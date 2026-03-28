# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import h5netcdf
import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import GSIConventionalLexicon, GSISatelliteLexicon
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray


@dataclass
class _GSIAsyncTask:
    """Small helper struct for Async tasks"""

    datetime_file: datetime
    datetime_max: datetime
    datetime_min: datetime
    gsi_file_path: str
    gsi_modifier: Callable
    gsi_obs_name: str
    e2s_obs_name: str
    satellite: str | None = None


class _MERRA21cObsBase:
    """Base class for MERRA-21c GSI data sources.

    This abstract base class provides common functionality for reading MERRA-21c
    GSI observation data from local filesystem.
    """

    SOURCE_ID: str  # To be defined by subclasses
    SCHEMA: pa.Schema  # To be defined by subclasses

    def __init__(
        self,
        base_path: str,
        experiment_id: str,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        max_workers: int = 24,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        base_path : str
            Base directory path for MERRA-21c data
            Example: "/discover/nobackup/projects/gmao/merra21c/TSE_staging"
        experiment_id : str
            Experiment identifier used in directory and file naming
            Example: "e5303_m21c_jan18"
        time_tolerance : TimeTolerance, optional
            Time tolerance window for filtering observations, by default np.timedelta64(10, 'm')
        max_workers : int, optional
            Max workers in async IO thread pool, by default 24
        cache : bool, optional
            Cache data source in local filesystem cache, by default True
        async_timeout : int, optional
            Time in seconds after which async fetch will be cancelled, by default 600
        verbose : bool, optional
            Log basic progress information, by default True
        """
        self.base_path = pathlib.Path(base_path)
        self.experiment_id = experiment_id
        self.obs_type = "ges"
        self._verbose = verbose
        self._cache = cache
        self._max_workers = max_workers
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

        # Verify base path exists
        if not self.base_path.exists():
            raise FileNotFoundError(
                f"Base path does not exist: {self.base_path}"
            )
    def _build_file_path(
        self,
        dt: datetime,
        gsi_platform: str,
        gsi_sensor: str,
        gsi_product: str,
    ) -> str:
        """Build local filesystem path for MERRA-21c GSI diagnostic file.

        Parameters
        ----------
        dt : datetime
            Datetime of the observation file
        gsi_platform : str
            GSI platform identifier (e.g., 't', 'uv', 'n20', 'npp')
        gsi_sensor : str
            GSI sensor identifier (e.g., 'conv', 'atms', 'amsua')
        gsi_product : str
            GSI product identifier (e.g., 'ges')

        Returns
        -------
        str
            Full path to the diagnostic file

        Example
        -------
        For datetime 2022-01-01 00:00:00 with platform='n20', sensor='atms':
        /discover/nobackup/projects/gmao/merra21c/TSE_staging/e5303_m21c_jan18/archive/obs/Y2022/M01/D01/H00/e5303_m21c_jan18.diag_n20_atms_ges.20220101_00z.nc4
        """
        year_dir = f"Y{dt.year:04d}"
        month_dir = f"M{dt.month:02d}"
        day_dir = f"D{dt.day:02d}"
        hour_dir = f"H{dt.hour:02d}"
        
        datetime_str = dt.strftime("%Y%m%d_%Hz")
        
        # Build filename: {exp_id}.diag_{platform}_{sensor}_{product}.{datetime}.nc4
        # Note: For satellites, platform is the satellite (n20, npp, etc.)
        #       For conventional, platform is the obs type (t, uv, etc.)
        filename = f"{self.experiment_id}.diag_{gsi_sensor}_{gsi_platform}_{gsi_product}.{datetime_str}.nc4"
    # GST    filename = f"{self.experiment_id}.diag_{gsi_platform}_{gsi_sensor}_{gsi_product}.{datetime_str}.nc4"
    # GST       filename = f"{self.experiment_id}.diag_{gsi_sensor}_{gsi_platform}_{gsi_product}.{datetime_str}.nc4"
    
        # Build full path
        file_path = (
            self.base_path 
            / self.experiment_id 
            / "archive" 
            / "obs" 
            / year_dir 
            / month_dir 
            / day_dir 
            / hour_dir 
            / filename
        )
    
        return str(file_path)
    
    def GT_build_file_path(
        self,
        dt: datetime,
        gsi_platform: str,
        gsi_sensor: str,
        gsi_product: str,
    ) -> str:
        """Build local filesystem path for MERRA-21c GSI diagnostic file.

        Parameters
        ----------
        dt : datetime
            Datetime of the observation file
        gsi_platform : str
            GSI platform identifier (e.g., 't', 'uv', 'amsua')
        gsi_sensor : str
            GSI sensor identifier (e.g., 'conv', 'n20')
        gsi_product : str
            GSI product identifier (e.g., 'ges')

        Returns
        -------
        str
            Full path to the diagnostic file

        Example
        -------
        For datetime 2022-01-01 00:00:00:
        /discover/nobackup/projects/gmao/merra21c/TSE_staging/e5303_m21c_jan18/archive/obs/Y2022/M01/D01/H00/e5303_m21c_jan18.diag_conv_t_ges.20220101_00z.nc4
        """
        year_dir = f"Y{dt.year:04d}"
        month_dir = f"M{dt.month:02d}"
        day_dir = f"D{dt.day:02d}"
        hour_dir = f"H{dt.hour:02d}"
        
        datetime_str = dt.strftime("%Y%m%d_%Hz")
        
        # Build filename: {exp_id}.diag_{sensor}_{platform}_{product}.{datetime}.nc4
 #GST       filename = f"{self.experiment_id}.diag_{gsi_platform}_{gsi_sensor}_{gsi_product}.{datetime_str}.nc4"
        filename = f"{self.experiment_id}.diag_{gsi_sensor}_{gsi_platform}_{gsi_product}.{datetime_str}.nc4"
        
        # Build full path
        file_path = (
            self.base_path 
            / self.experiment_id 
            / "archive" 
            / "obs" 
            / year_dir 
            / month_dir 
            / day_dir 
            / hour_dir 
            / filename
        )
        
        return str(file_path)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch observations for a set of timestamps.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            DataFrame column names to return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        df = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, variable, fields), timeout=self.async_timeout
            )
        )

        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data."""
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        print("time_list variable_list", time_list, variable_list)
        async_tasks = self._create_tasks(time_list, variable_list)
        file_path_set = {task.gsi_file_path for task in async_tasks}
        print("file_path_set", file_path_set)
        fetch_jobs = [self._fetch_local_file(path) for path in file_path_set]
        await tqdm.gather(
            *fetch_jobs, desc="Loading GSI files", disable=(not self._verbose)
        )

        df = self._compile_dataframe(async_tasks, variable_list, schema)

        return df

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        """Create async tasks for fetching data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_tasks.")

    async def _fetch_local_file(self, path: str) -> None:
        """Copy local file into cache if needed.

        Parameters
        ----------
        path : str
            Local filesystem path to copy
        """
        cache_path = self.cache_path(path)
        if not pathlib.Path(cache_path).is_file():
            source_path = pathlib.Path(path)
            print(source_path)
            if not source_path.is_file():
                self._handle_missing_file(path)
                return
            
            try:
                # Use async file operations for consistency
                await asyncio.get_event_loop().run_in_executor(
                    None, shutil.copy2, str(source_path), cache_path
                )
            except Exception as e:
                logger.error(f"Failed to copy {path}: {e}")
                self._handle_missing_file(path)

    def _handle_missing_file(self, path: str) -> None:
        """Handle missing file during fetch. Can be overridden by subclasses."""
        logger.error(f"File {path} not found")
        raise FileNotFoundError(f"File {path} not found")

    def _compile_dataframe(
        self,
        async_tasks: list[_GSIAsyncTask],
        variables: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Compile fetched data into a DataFrame."""
        frames: list[pd.DataFrame] = []
        missing_files = []
        empty_after_filter = []
        
        for task in async_tasks:
            # Overwrite obs column name (needed for uv)
            column_map = self._build_column_map(schema)
            column_map[task.gsi_obs_name] = "observation"
            local_path = self.cache_path(task.gsi_file_path)
            
            if not pathlib.Path(local_path).is_file():
                logger.warning("Cached file missing for {}", task.gsi_file_path)
                missing_files.append(task.gsi_file_path)
                continue
                
            try:
                with h5netcdf.File(local_path, "r") as ds:
                    data: dict[str, np.ndarray] = {}
                    for name, dset in ds.variables.items():
                        if name not in column_map:
                            continue
                        values = np.asarray(dset[:])
                        pa_type = self.SCHEMA.field(column_map[name]).type
                        # Convert char arrays into strings for DF
                        if values.dtype.kind == "S" and values.ndim == 2:
                            values = values.view(f"S{values.shape[1]}").ravel()
                            values = np.char.rstrip(
                                np.char.decode(values, "utf-8"), "\x00"
                            )
                        # Apply subclass-specific transformations
                        values = self._transform_column(name, values, task, ds)
                        data[name] = pa.array(values, type=pa_type)
                        
                    df = pd.DataFrame(data)
                    logger.debug(f"Loaded {len(df)} rows from {task.gsi_file_path}")
                    
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to read {}: {}", local_path, exc)
                raise exc

            # Rename columns
            df.rename(columns=column_map, inplace=True)
            # Add e2s columns
            df["variable"] = task.e2s_obs_name
            df.attrs["source"] = self.SOURCE_ID
            self._add_task_columns(df, task)

            # Filter by time window
            logger.debug(f"Filtering time window: {task.datetime_min} to {task.datetime_max}")
            logger.debug(f"Time range in data: {df['time'].min()} to {df['time'].max()}")
            
            mask = (df["time"] >= task.datetime_min) & (df["time"] <= task.datetime_max)
            df_filtered = df.loc[mask]
            
            logger.debug(f"After time filter: {len(df_filtered)} rows (from {len(df)})")
            
            if len(df_filtered) == 0:
                empty_after_filter.append(task.gsi_file_path)
                continue
                
            frames.append(task.gsi_modifier(df_filtered))

        # Report summary
        logger.info(f"Processed {len(async_tasks)} tasks:")
        logger.info(f"  - {len(missing_files)} files not found")
        logger.info(f"  - {empty_after_filter} files with no data after time filtering")
        logger.info(f"  - {len(frames)} frames with data")

        # Handle case where no data was found
        if not frames:
            logger.warning("No observation data found for the requested time/variables")
            # Return empty DataFrame with correct schema
            empty_data = {}
            for field in schema:
                # Get pandas dtype from PyArrow type
                if field.type == pa.timestamp("ns"):
                    dtype = 'datetime64[ns]'
                elif field.type == pa.float32():
                    dtype = 'float32'
                elif field.type == pa.uint16():
                    dtype = 'uint16'
                elif field.type == pa.string():
                    dtype = 'object'
                else:
                    dtype = 'object'
                empty_data[field.name] = pd.Series(dtype=dtype)
            return pd.DataFrame(empty_data)

        result = pd.concat(frames, ignore_index=True)
        return result[[name for name in schema.names if name in result.columns]]  
    def _build_column_map(self, schema: pa.Schema) -> dict[str, str]:
        """Build mapping from GSI column names to schema column names."""
        column_map = {}
        for field in schema:
            if field.metadata is None or b"gsi_name" not in field.metadata:
                continue
            column_map[field.metadata[b"gsi_name"].decode("utf-8")] = field.name
        # Always include time field for filtering
        time_field = self.SCHEMA.field("time")
        column_map[time_field.metadata[b"gsi_name"].decode("utf-8")] = time_field.name
        return column_map

    def _transform_column(
        self,
        name: str,
        values: np.ndarray,
        task: _GSIAsyncTask,
        ds: h5netcdf.File,
    ) -> np.ndarray:
        """Transform column values. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _transform_column.")

    def _add_task_columns(self, df: pd.DataFrame, task: _GSIAsyncTask) -> None:
        """Add task-specific columns to DataFrame. Override in subclasses."""
        pass

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert fields parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field specification. Can be:
            - None: Returns the full class SCHEMA
            - str: Single field name to select from SCHEMA
            - list[str]: List of field names to select from SCHEMA
            - pa.Schema: Validated against class SCHEMA for compatibility

        Returns
        -------
        pa.Schema
            A PyArrow schema containing only the requested fields

        Raises
        ------
        KeyError
            If a requested field name is not found in the class SCHEMA
        TypeError
            If a field type in the provided schema doesn't match the class SCHEMA
        ValueError
            If required fields are missing
        """
        if fields is None:
            return cls.SCHEMA

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
            # Validate provided schema against class schema
            for field in fields:
                if field.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{field.name}' not found in class SCHEMA. "
                        f"Available fields: {cls.SCHEMA.names}"
                    )
                expected_type = cls.SCHEMA.field(field.name).type
                if field.type != expected_type:
                    raise TypeError(
                        f"Field '{field.name}' has type {field.type}, "
                        f"expected {expected_type} from class SCHEMA"
                    )
            return fields

        # fields is list[str] - select fields from class schema
        selected_fields = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not found in class SCHEMA. "
                    f"Available fields: {cls.SCHEMA.names}"
                )
            selected_fields.append(cls.SCHEMA.field(name))

        return pa.schema(selected_fields)

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for GSI based on offline knowledge

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            start_date = datetime(2020, 1, 1)  # Adjust based on MERRA-21c availability
            if time < start_date:
                raise ValueError(
                    f"Requested date time {time} needs to be after {start_date} for MERRA-21c observations"
                )

    def cache_path(self, path: str) -> str:
        """Gets local cache path given filesystem path

        Parameters
        ----------
        path : str
            Local filesystem path

        Returns
        -------
        str
            Local path of cached file
        """
        sha = hashlib.sha256(path.encode())
        filename = sha.hexdigest()
        return os.path.join(self.cache, filename)

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "merra21c_gsi")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_gsi_{self._tmp_cache_hash}"
            )
        return cache_location


class MERRA21cObsConv(_MERRA21cObsBase):
    """MERRA-21c GSI conventional (in-situ) observations data

    Parameters
    ----------
    base_path : str
        Base directory path for MERRA-21c data
        Example: "/discover/nobackup/projects/gmao/merra21c/TSE_staging"
    experiment_id : str
        Experiment identifier used in directory and file naming
        Example: "e5303_m21c_jan18"
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric ± window) or a tuple (lower, upper) for asymmetric windows,
        by default, np.timedelta64(10, 'm').
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent file operations, by default 24.
    cache : bool, optional
        Cache data source in local filesystem cache, by default True.
    async_timeout : int, optional
        Time in seconds after which the async fetch will be cancelled if not finished,
        by default 600.
    verbose : bool, optional
        Log basic progress information, by default True.

    Note
    ----
    File path structure:
    {base_path}/{experiment_id}/archive/obs/Y{YYYY}/M{MM}/D{DD}/H{HH}/{experiment_id}.diag_conv_{variable}_ges.{YYYYMMDD_HHz}.nc4

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        ds = MERRA21cObsConv(
            base_path="/discover/nobackup/projects/gmao/merra21c/TSE_staging",
            experiment_id="e5303_m21c_jan18",
            time_tolerance=timedelta(hours=2)
        )
        df = ds(datetime(2022, 1, 1, 0), ["u"])
    """

    SOURCE_ID = "earth2studio.data.MERRA21cObsConv"
    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns"), metadata={"gsi_name": "Time"}),
            pa.field(
                "pres", pa.float32(), nullable=True, metadata={"gsi_name": "Pressure"}
            ),
            pa.field(
                "elev", pa.float32(), nullable=True, metadata={"gsi_name": "Height"}
            ),
            pa.field(
                "type",
                pa.uint16(),
                nullable=True,
                metadata={"gsi_name": "Observation_Type"},
            ),
            pa.field(
                "class",
                pa.string(),
                nullable=True,
                metadata={"gsi_name": "Observation_Class"},
            ),
            pa.field("lat", pa.float32(), metadata={"gsi_name": "Latitude"}),
            pa.field("lon", pa.float32(), metadata={"gsi_name": "Longitude"}),
            pa.field("station", pa.string(), metadata={"gsi_name": "Station_ID"}),
            pa.field(
                "station_elev",
                pa.float32(),
                nullable=True,
                metadata={"gsi_name": "Station_Elevation"},
            ),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        tasks: list[_GSIAsyncTask] = []
        for v in variable:
            try:
                gsi_name, modifier = GSIConventionalLexicon[v]  # type: ignore
                gsi_platform, gsi_sensor, gsi_product, gsi_name = gsi_name.split("::")
            except KeyError:
                if v in GSISatelliteLexicon:
                    logger.warning(
                        f"Variable id {v} is a satellite variable, skipping in conventional fetch"
                    )
                    continue
                logger.error(f"Variable id {v} not found in GSI lexicon")
                raise

            for t in time_list:
                tmin = t + self._tolerance_lower
                tmax = t + self._tolerance_upper
                day = tmin.replace(minute=0, second=0, microsecond=0)
                day = day.replace(hour=(day.hour // 6) * 6)
                while day <= tmax:
                    file_path = self._build_file_path(
                        day, gsi_sensor, gsi_platform, gsi_product
                        # day, gsi_platform, gsi_sensor, gsi_product
                    )
                    tasks.append(
                        _GSIAsyncTask(
                            datetime_file=day,
                            datetime_min=tmin,
                            datetime_max=tmax,
                            gsi_file_path=file_path,
                            gsi_modifier=modifier,
                            gsi_obs_name=gsi_name,
                            e2s_obs_name=v,
                        )
                    )
                    day = day + timedelta(hours=6)
        return tasks 
       
    def _transform_column(
        self,
        name: str,
        values: np.ndarray,
        task: _GSIAsyncTask,
        ds: h5netcdf.File,
    ) -> np.ndarray:
        """Transform column values for conventional data."""
        # Convert hours offset to timedelta, and add to datetime of file
        if name == "Time":
            values = pd.to_timedelta(values, unit="h") + task.datetime_file
        return values

    def _build_column_map(self, schema: pa.Schema) -> dict[str, str]:
        """Build column map including elev field required for modifiers."""
        column_map = super()._build_column_map(schema)
        # Required for modifier filtering
        elev_field = self.SCHEMA.field("elev")
        column_map[elev_field.metadata[b"gsi_name"].decode("utf-8")] = elev_field.name
        return column_map


class MERRA21cObsSat(_MERRA21cObsBase):
    """MERRA-21c GSI satellite observations data

    Parameters
    ----------
    base_path : str
        Base directory path for MERRA-21c data
        Example: "/discover/nobackup/projects/gmao/merra21c/TSE_staging"
    experiment_id : str
        Experiment identifier used in directory and file naming
        Example: "e5303_m21c_jan18"
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric ± window) or a tuple (lower, upper) for asymmetric windows,
        by default, np.timedelta64(10, 'm').
    satellites : list[str], optional
        List of satellite platforms to include, by default includes all platforms.
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent file operations, by default 24.
    cache : bool, optional
        Cache data source in local filesystem cache, by default True.
    async_timeout : int, optional
        Time in seconds after which the async fetch will be cancelled if not finished,
        by default 600.
    verbose : bool, optional
        Log basic progress information, by default True.

    Note
    ----
    File path structure:
    {base_path}/{experiment_id}/archive/obs/Y{YYYY}/M{MM}/D{DD}/H{HH}/{experiment_id}.diag_{sensor}_{satellite}_ges.{YYYYMMDD_HHz}.nc4

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        # Use all possible satellites
        ds = MERRA21cObsSat(
            base_path="/discover/nobackup/projects/gmao/merra21c/TSE_staging",
            experiment_id="e5303_m21c_jan18",
            time_tolerance=timedelta(hours=2)
        )
        df = ds(datetime(2022, 1, 1, 0), ["atms"])

        # Use specific satellite
        ds = MERRA21cObsSat(
            base_path="/discover/nobackup/projects/gmao/merra21c/TSE_staging",
            experiment_id="e5303_m21c_jan18",
            time_tolerance=timedelta(hours=2),
            satellites=["n20"]
        )
        df = ds(datetime(2022, 1, 1, 0), ["atms"])
    """

    SOURCE_ID = "earth2studio.data.MERRA21cObsSat"
    VALID_SATELLITES = frozenset(
        [
            "npp",
            "metop-a",
            "metop-b",
            "metop-c",
            "n15",
            "n16",
            "n17",
            "n18",
            "n19",
            "n20",
        ]
    )
    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns"), metadata={"gsi_name": "Obs_Time"}),
            pa.field(
                "elev", pa.float32(), nullable=True, metadata={"gsi_name": "Elevation"}
            ),
            pa.field(
                "class",
                pa.string(),
                nullable=True,
                metadata={"gsi_name": "Observation_Class"},
            ),
            pa.field("lat", pa.float32(), metadata={"gsi_name": "Latitude"}),
            pa.field("lon", pa.float32(), metadata={"gsi_name": "Longitude"}),
            pa.field("scan_angle", pa.float32(), metadata={"gsi_name": "Scan_Angle"}),
            pa.field(
                "channel_index",
                pa.uint16(),
                nullable=True,
                metadata={"gsi_name": "Channel_Index"},
            ),
            pa.field("solza", pa.float32(), metadata={"gsi_name": "Sol_Zenith_Angle"}),
            pa.field(
                "solaza", pa.float32(), metadata={"gsi_name": "Sol_Azimuth_Angle"}
            ),
            pa.field(
                "satellite_za", pa.float32(), metadata={"gsi_name": "Sat_Zenith_Angle"}
            ),
            pa.field(
                "satellite_aza",
                pa.float32(),
                metadata={"gsi_name": "Sat_Azimuth_Angle"},
            ),
            pa.field("satellite", pa.string()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def __init__(
        self,
        base_path: str,
        experiment_id: str,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        satellites: list[str] | None = None,
        max_workers: int = 24,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        if satellites is None:
            satellites = list(self.VALID_SATELLITES)
        else:
            invalid = set(satellites) - self.VALID_SATELLITES
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {invalid}. "
                    f"Valid satellites are: {sorted(self.VALID_SATELLITES)}"
                )
        self.satellites = satellites
        super().__init__(
            base_path=base_path,
            experiment_id=experiment_id,
            time_tolerance=time_tolerance,
            max_workers=max_workers,
            cache=cache,
            async_timeout=async_timeout,
            verbose=verbose,
        )

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        tasks: list[_GSIAsyncTask] = []
        for v in variable:
            try:
                gsi_name, modifier = GSISatelliteLexicon[v]  # type: ignore
                gsi_platforms0, gsi_sensor, gsi_product, gsi_name = gsi_name.split("::")
                gsi_platforms = [
                    p for p in gsi_platforms0.split(",") if p in self.satellites
                ]
            except KeyError:
                if v in GSIConventionalLexicon:
                    logger.warning(
                        f"Variable id {v} is a UFS conventional variable, skipping in satellite fetch"
                    )
                    continue
                logger.error(f"Variable id {v} not found in GSI lexicon")
                raise

            for gsi_platform in gsi_platforms:
                for t in time_list:
                    tmin = t + self._tolerance_lower
                    tmax = t + self._tolerance_upper
                    day = tmin.replace(minute=0, second=0, microsecond=0)
                    day = day.replace(hour=(day.hour // 6) * 6)
                    while day <= tmax:
                        file_path = self._build_file_path(
                            day, gsi_platform, gsi_sensor, gsi_product
                        )
                        tasks.append(
                            _GSIAsyncTask(
                                datetime_file=day,
                                datetime_min=tmin,
                                datetime_max=tmax,
                                gsi_file_path=file_path,
                                gsi_modifier=modifier,
                                gsi_obs_name=gsi_name,
                                e2s_obs_name=v,
                                satellite=gsi_platform,
                            )
                        )
                        day = day + timedelta(hours=6)
        return tasks

    def _handle_missing_file(self, path: str) -> None:
        """Satellite data may have missing platforms, just warn instead of error."""
        logger.warning(f"File {path} not found")

    def _transform_column(
        self,
        name: str,
        values: np.ndarray,
        task: _GSIAsyncTask,
        ds: h5netcdf.File,
    ) -> np.ndarray:
        """Transform column values for satellite data."""
        # Convert hours offset to timedelta, and add to datetime of file
        if name == "Obs_Time":
            values = pd.to_timedelta(values, unit="h") + task.datetime_file
        # Channel index actually seems to be a pointer to sensor channels
        if name == "Channel_Index":
            sensor_chan = ds["sensor_chan"][:].astype(np.uint16)
            values = sensor_chan[values.astype(np.uint16) - 1]
        return values

    def _add_task_columns(self, df: pd.DataFrame, task: _GSIAsyncTask) -> None:
        """Add satellite column for satellite data."""
        df["satellite"] = task.satellite


