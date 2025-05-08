import glob
import xarray as xr
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from earthnet.util import instantiate_from_config

import matplotlib.pyplot as plt
import pickle

class EarthNetRunner(object):
    def __init__(self, config_file):
        self.config = OmegaConf.load(config_file)
        self.model = instantiate_from_config(self.config.model)

    def prepare_inputs(self, data):
        assert len(data.time) == 12

        # del data["PIce"]
        for v in data.data_vars:
            data[v] = data[v].astype("float16")
        # data.to_zarr("earthmae_abi-viirs-atms_inputs.zarr")

        # data = data.isel(lat=slice(0,500), lon=slice(0,500))
        inputs = dict()
        for d, params in self.config.data.train.params.domains.items():
            inputs[d] = []
            for v in params["vars"]:
                key = d + "_" + v
                if len(data[key].dims) == 4:
                    c_dim = np.setdiff1d(data[key].dims, ["time", "lat", "lon"])[0]
                    data[key] = data[key].transpose(c_dim, "time", "lat", "lon")
                elif len(data[key].dims) == 3:
                    if "time" in data[key].dims:
                        new_dim = "band_" + v
                        data[key] = data[key].expand_dims(new_dim)
                elif (len(data[key].dims) == 2) and (
                    "time" not in data[key].dims
                ):  # make 2d sample (C, H, W)
                    new_dim = "band_" + v
                    # x[key] = x[key].expand_dims("time_0")
                    data[key] = data[key].expand_dims(new_dim)

                inputs[d].append(torch.Tensor(data[key].values))

            inputs[d] = torch.cat(
                inputs[d]
            )  # torch.cat([torch.Tensor(data[v].values) for v in params["vars"]], 1)

            inputs[d] = inputs[d].unsqueeze(0)
        return inputs

    def forward(self, x: dict) -> dict:
        """
        Args:
            x: dict of modalities. eg. {'goes16': torch.Tensor(), 'viirs': ....}

        Returns:
            Dict of gap-filled modalities
        """
        patch_size = self.config.model.params.img_size
        patch_overlap = patch_size // 4
        trim = 0

        device = self.model.device

        # x = self.model.data_transform(x)

        # perform inference on patches
        counters = {k: np.zeros(v.shape, dtype=np.float32) for k, v in x.items()}
        res_sum = {k: np.zeros(v.shape, dtype=np.float32) for k, v in x.items()}
        if "goes16" in x.keys():
            height, width = x["goes16"].shape[-2:]
        else:
            height, width = x[list(x.keys())[0]].shape[-2:]

        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7067425/
        # hann = 1/4 * (1 - cos(2 \pi i / I)) * (1 - cos(2 \pi j / J))
        xr = np.arange(0, patch_size, dtype=np.float32)
        xr_w = 1 - np.cos(2 * np.pi * xr / patch_size)
        pdf = np.outer(xr_w, xr_w) / 4

        border_mask = np.zeros(pdf.shape)
        if trim == 0:
            border_mask = 1.0
        else:
            border_mask[trim:-trim, trim:-trim] = 1
        pdf *= border_mask

        indices = []
        ix_iy = [
            (ix, iy)
            for ix in range(0, height, patch_size - patch_overlap)
            for iy in range(0, width, patch_size - patch_overlap)
        ]
        for ix, iy in tqdm(ix_iy):
            ix = min(ix, height - patch_size)
            iy = min(iy, width - patch_size)
            if (ix, iy) in indices:
                continue
            indices.append((ix, iy))
            patch_inputs = {
                k: v[..., ix : ix + patch_size, iy : iy + patch_size].to(device)
                for k, v in x.items()
            }

            patch_inputs = self.model.data_transform(patch_inputs)

            patch_outputs = self.model(patch_inputs)[0]
            # patch_outputs = self.model(patch_inputs, mask_inputs=True)[0]
            patch_outputs = self.model.backward_transform(patch_outputs)

            for k, v in patch_outputs.items():
                res_sum[k][..., ix : ix + patch_size, iy : iy + patch_size] += (
                    v.cpu().detach().numpy() * pdf
                )
                counters[k][..., ix : ix + patch_size, iy : iy + patch_size] += pdf

            del patch_inputs

        out = {}
        for k, v in res_sum.items():
            out[k] = v / counters[k]  # .compute()
            # print(k, out[k].shape)

        return out

    def outputs_to_dataset(self, ds: xr.Dataset, outputs: dict) -> xr.Dataset:
        """
        Returns a gap-filled dataset like ds
        """
        pred = xr.full_like(ds, np.nan)
        for d, params in self.config.data.train.params.domains.items():
            idx = 0
            for v in params["vars"]:
                k = d + "_" + v

                dim_diff = np.setdiff1d(pred[k].dims, ["time", "lat", "lon"])
                if len(dim_diff) == 1:
                    idx_end = pred[dim_diff[0]].shape[0] + idx
                    pred[k] = pred[k].transpose(
                        dim_diff[0], "time", "lat", "lon", missing_dims="ignore"
                    )
                    pred[k].values = outputs[d][0, idx:idx_end]
                    idx = idx_end
                elif len(dim_diff) == 0:
                    pred[k].values = outputs[d][0, idx]
                    idx += 1
                else:
                    raise ValueError()

                # print(k, idx, idx_end, dim_diff)

                pred[k] = pred[k].astype("float16")
                idx_end = idx

        return pred
if __name__ == "__main__":
    config_file = "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMZeusAI/earthnet/multimodal_earth.yaml"

    config = OmegaConf.load(config_file)
    print("Loading config: ", config)

    runner = EarthNetRunner(config_file)

    print("Loading data")
    data_files = glob.glob(config.data.test.params.data_dir + "*.zarr")
    print("Loading data_files: ", data_files)
    data = xr.open_zarr(data_files[0])
    data = data.isel(time=slice(0, 12))

    print("Running model: data= ", data)
    inputs = runner.prepare_inputs(data)
    print(inputs.keys())

    # Filter inputs you'd like to ignore

    #inputs['goes16'][:] = np.nan
    #inputs['goes18'][:] = np.nan
    #inputs['gk2a'][:] = np.nan
    #inputs['seviri'][:] = np.nan
    #inputs['atms'][:] = np.nan
    #inputs['srtm'][:] = np.nan
    #inputs['viirs'][:] = np.nan

    for k, v in inputs.items():
    #    Remove last timestep
    #    if len(v.shape) == 5:
    #        print(k, v.shape)
    #        inputs[k][:,:,-1:] = np.nan

        # Remove MIRS data from infernece
        if 'mirs' in k:
            print(k)
            inputs[k][:] = np.nan

    outputs = runner.forward(inputs)
    pred = runner.outputs_to_dataset(data, outputs)
    pred['mirs_snd_temp_PTemp'] = pred['mirs_snd_temp_PTemp'].astype(np.float32)

    print("Storing pred: ", pred)
    # Save the results to a pickle file
    with open(f'pred.pkl', 'wb') as f:
        pickle.dump(pred, f)
    print("After storing pred: ")
 
    # GOES
    # diff = pred['goes18_Rad'].isel(time=6, goes18_band=9) - pred['goes16_Rad'].isel(time=6, goes16_band=9)
    goes_band = 14

    fig, axs = plt.subplots(1,2,figsize=(12,4))
    axs = axs.flatten()

    ds = data
    ds['goes16_Rad'].sel(goes16_band=goes_band).isel(time=11).plot(ax=axs[0])
    axs[0].set_title("GOES16 Obs")

    pred['goes16_Rad'].sel(goes16_band=goes_band).isel(time=11).plot(ax=axs[1])
    axs[1].set_title("GOES16 Prediction")

    plt.show()

    diff = (pred['goes16_Rad'] - ds['goes16_Rad'])
    diff.sel(goes16_band=goes_band).mean('time').plot()
    plt.title("Mean Error")
    plt.show()

    flat = diff.values.flatten()
    flat = flat[np.isfinite(flat)]

    bias = np.mean(flat)
    mae = np.mean(np.abs(flat))

    print(f"Bias: {bias}, MAE: {mae}")

    # Save the plot as a PNG image
    plt.savefig("goes16.png")

    # Save the plot with higher DPI and tight bounding box
    plt.savefig("goes16_highres.png", dpi=300, bbox_inches='tight')

    # Save the plot as a PDF
    #plt.savefig("my_plot.pdf")

    plt.close()

    #print("Prediction to netcdf:\n", pred)
    #pred.to_netcdf("earthnet_predictions.nc")

    #print("Prediction to zarr:\n", pred)
    #pred.to_zarr("earthnet_predictions.zarr")

