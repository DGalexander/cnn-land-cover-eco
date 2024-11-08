import sys
import os
#sys.path.append('/home/jovyan/work/cnn-land-cover/scripts/')
sys.path.append('../cnn-land-cover/scripts/')
import land_cover_models as lcm
import rioxarray as rio
import torch
import xarray as xr
import numpy as np
import pytorch_lightning as pl
from geocube.api.core import make_geocube
import geopandas as gpd
import pickle
import rasterio
import glob
from tqdm.autonotebook import tqdm
import gc
import shapely.geometry as sgeom

# Some hardcoded class mapping information
##ROOT = '/home/jovyan/work/cnn-land-cover/content/label_mapping_dicts'
ROOT = '../cnn-land-cover/content/label_mapping_dicts'
MD_PATHS = {
    'main': os.path.join(
        ROOT,
        'label_mapping_dict__main_categories_F3inDE_noFGH__2023-04-21-1315.pkl'
    ),
    'all': os.path.join(
        ROOT,
        'label_mapping_dict__all_relevant_subclasses__2023-04-20-1540.pkl'
    ),
    'c': os.path.join(
        ROOT,
        'label_mapping_dict__C_subclasses_only__2023-04-20-1540.pkl'
    ),
    'd': os.path.join(
        ROOT,
        'label_mapping_dict__D_subclasses_only__2023-04-20-1540.pkl'
    ),
    'e': os.path.join(
        ROOT,
        'label_mapping_dict__E_subclasses_and_F3d_only__2023-04-20-1541.pkl'
    ),
}

def load_patches_dataset(path, map_dict, file_list=None, proc_fn=None):
    dsp = lcm.DataSetPatches(
        im_dir=os.path.join(path, 'images'),
        mask_dir=os.path.join(path, 'masks'),
        mask_suffix='_lc_2022_detailed_mask.npy',
        mask_dir_name='masks',
        list_tile_patches_use=file_list,
        preprocessing_func=proc_fn,
        shuffle_order_patches=False, # Check this!
        relabel_masks=True,
        subsample_patches=False,
        path_mapping_dict=map_dict,
        random_transform_data=False
    )
    dsp.remove_no_class_patches()
    
    return dsp

def get_buffered_tile(vrt, bbox):
    
    aoi = rio.open_rasterio(
        vrt,
        masked=True, # Nodata is nan
        cache=False
    ).rio.clip_box(*bbox)
    
    return aoi

def open_tile(fname, vrt, overlap=0):
    """Return tile from VRT with pixel overlap.
    
    Used to read in overlapping tiles based on
    VRT images (multiple files within
    a folder).
    """
    tile = rio.open_rasterio(
        fname,
        masked=True, # Nodata is nan
        cache=False
    )
    
    bounds = np.asarray(tile.rio.bounds()) # left, bottom, right, top
    resolution = np.asarray(tile.rio.resolution())[::-1]
    bbox = np.zeros(bounds.shape)
    
    # Add overlap to bounds
    bbox[::2] = bounds[::2] + overlap*resolution
    bbox[1::2] = bounds[1::2] + overlap*resolution
    
    vrt_tile = get_buffered_tile(vrt, bbox=bbox)
    
    return vrt_tile

class TiledImageRio():
    """Whole image dataset for inference.
    
    Uses xarray instead of GDAL"""
    def __init__(self, image_rio, tile_size_r_c, overlap=0, prep_func=None, ol_added=False):
        
        super(TiledImageRio, self).__init__()
        
        self.src = image_rio
        self.bc, self.rc, self.cc = image_rio.shape
    
        self.tile_h, self.tile_w = tile_size_r_c
        self.ol = overlap

        self._tile_image()
        self._pad_tiles()

        self.buffered = ol_added
        
        if prep_func is not None:
            rgb_means = prep_func.keywords['mean']
            rgb_std = prep_func.keywords['std']

            rgb_means = torch.tensor(np.array(rgb_means)[:, None, None])
            rgb_std = torch.tensor(np.array(rgb_std)[:, None, None])

            dtype = torch.float32
            
            self.rgb_means = rgb_means.type(dtype)
            self.rgb_std = rgb_std.type(dtype)
            
            self.preprocess = self.zscore_image
        else:
            self.preprocess = self.pass_image
        
    def get_writer_callback(self, tile_name, output_dir):
        pred_writer = TiledImageWriter(
            src=self.src,
            tile_name=tile_name,
            overlap=self.ol,
            tile_map=self.tile_map,
            output_dir=output_dir,
            write_interval='epoch',
            input_buffered=self.buffered
        )
        
        return pred_writer
    
    def __len__(self):
        
        return len(self.tile_map)
    
    def __getitem__(self, index):
        """Return image patch"""
        xoff, yoff, xcount, ycount = self.overlaps[index, 1:]
    
        pix = self.src[:, yoff:yoff+ycount, xoff:xoff+xcount].values
        padded = np.pad(pix[:3].transpose(1, 2, 0), self.padding[index], 'reflect')
        padded = torch.as_tensor(padded.transpose(2, 0, 1)).float()
         
        return [self.preprocess(padded)]
    
    def zscore_image(self, im):
        im = im / 255 
        im = (im - self.rgb_means) / self.rgb_std
        return im

    def get_image_dimensions(self):
        return self.bc, self.rc, self.cc
    
    def pass_image(self, im):
        return im
    
    def _pad_tiles(self):
        ol = self.ol
        tile_h, tile_w = self.tile_h, self.tile_w
        tiles = self.tile_map
        overlaped = tiles + np.array([0, -ol, -ol, ol*2, ol*2]) 
        i, c, r, w, h = overlaped.transpose(1, 0)
        
        rc, cc = self.rc, self.cc
        
        padding = np.zeros(len(i)*6, dtype=int).reshape(len(i), 3, 2)
        
        # Edge cases
        padding[i[r<0], 0, 0] = tiles[i[r<0], 2] + ol
        overlaped[i[r<0], 4] -= padding[i[r<0], 0, 0]
        overlaped[i[r<0], 2] = tiles[i[r<0], 2]
        
        padding[i[c<0], 1, 0] = tiles[i[c<0], 1] + ol
        overlaped[i[c<0], 3] -= padding[i[c<0], 1, 0]
        overlaped[i[c<0], 1] = tiles[i[c<0], 1]
        
        padding[i[r+h>rc], 0, 1] = tile_h+ol - tiles[i[r+h>rc], 4]
        r_off = tiles[i[r+h>rc], 2] - overlaped[i[r+h>rc], 2]
        overlaped[i[r+h>rc], 4] = tiles[i[r+h>rc], 4] + r_off
        
        padding[i[c+w>cc], 1, 1] = tile_w+ol - tiles[i[c+w>cc], 3]
        c_off = tiles[i[c+w>cc], 1] - overlaped[i[c+w>cc], 1]
        overlaped[i[c+w>cc], 3] = tiles[i[c+w>cc], 3] + c_off
        
        self.padding = padding
        self.overlaps = overlaped
        
    def _tile_image(self):
        ol = self.ol
        tile_h, tile_w = self.tile_h, self.tile_w
        rc, cc = self.rc, self.cc
        tiles = []

        for r in np.arange(0, rc, tile_h):
            for c in np.arange(0, cc, tile_w):
                # Long hand
                if rc-r < tile_h:
                    h = rc-r
                else:
                    h = tile_h
                    
                if cc-c < tile_w:
                    w = cc-c
                else:
                    w = tile_w
                
                tiles.append([len(tiles), c, r, w, h])
                
        self.tile_map = np.array(tiles)

    def __len__(self):
        return len(self.tile_map)

class TiledImageWriter(pl.callbacks.BasePredictionWriter):
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html
    def __init__(
        self, src, tile_name, overlap, tile_map, output_dir, write_interval, input_buffered=False):
        
        super().__init__(write_interval)
        self.output_dir = output_dir
        
        self.ol = overlap 
        self.src_xcoords = src.coords['x'].values
        self.src_ycoords = src.coords['y'].values
        self.src_crs = src.rio.crs
        self.tile_map = tile_map
        self.out_dir = output_dir
        self.shape = src.shape
        self.tname = tile_name
        self.buffered = input_buffered
        
        self.leading = len(str(len(tile_map)))
        self.mask = False
        
    def _load_src(self, path):
        src = rio.open_rasterio(
            path,
            masked=True, # Nodata as nan
            cache=False,
            #chunks='auto' # Todo
        )
            
        return src
    
    def set_mask(self, mask_path, classes_to_keep):
        """Set mask tile and classes (to keep)"""
        self.mask_path = mask_path
        self.mask_cls = classes_to_keep
        self.mask = True

    def get_mask(self):
        """Mask classes (to keep)"""
        mask_tile = self._load_src(self.mask_path)
        mask = np.zeros(mask_tile.shape)
        
        for cls in self.mask_cls:
            mask[mask_tile==cls] = 1

        return mask[0] # Single dim

    def _write_rio(self, index, tile, fname=None):
        """Write a tile as a GTiff"""
        ol = self.ol
        xoff, yoff, xcount, ycount = self.tile_map[index, 1:]
  
        trimmed = tile.cpu()[ol:ol+ycount, ol:ol+xcount]
        
        dims = ['y', 'x']
        coords = [
            self.src_ycoords[yoff:yoff+ycount],
            self.src_xcoords[xoff:xoff+xcount]
        ]
        
        if fname is None: # write the tile index
            fname = str(index).rjust(self.leading, '0')+'.tif'
            
        outpath = os.path.join(self.out_dir, fname)
        
        raster = (
            xr.DataArray(trimmed, coords=coords, dims=dims)
            .astype('int32')
            .rio.write_crs(self.src_crs)
            .rio.to_raster(outpath)
        )
        
        
    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx
    ):
        max_pred = torch.argmax(prediction, dim=1)
        
        for i, pred in enumerate(max_pred): # i in batch
            self._write_rio(batch_indices[i], pred)
            
    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices
    ):
        # Initialize an output array
        nb, nr, nw = self.shape
        ol = self.ol
        out_pix = np.zeros((nr, nw))

        dataloader_index = 0 # Single input
        
        for batch_i, tile_indices in enumerate(batch_indices[dataloader_index]):
            preds = predictions[batch_i]
            max_preds = torch.argmax(preds, dim=1)
            
            for i, tile_index in enumerate(tile_indices):
                xoff, yoff, xcount, ycount = self.tile_map[tile_index, 1:]

                tile = max_preds[i]
                trimmed = tile.cpu()[ol:ol+ycount, ol:ol+xcount]
                
                out_pix[yoff:yoff+ycount, xoff:xoff+xcount] = trimmed

        dims = ['y', 'x']
        coords = [
            self.src_ycoords,
            self.src_xcoords
        ]
        
        fname = self.tname
        outpath = os.path.join(self.out_dir, fname)

        raster = (
            xr.DataArray(out_pix, coords=coords, dims=dims)
            .astype('int8')
            .rio.write_crs(self.src_crs)
        )

        if self.buffered:
            raster = raster[ol:-ol, ol:-ol]

        if self.mask: 
            mask = self.get_mask()
            raster = raster.where(mask==1, 0)
            
        raster.rio.to_raster(outpath)

class VrtTileClassifier():
    def __init__(self, model, vrt, tile_list, out_path, add_overlap=True):
        self.overlap = 64
        self.tiley, self.tilex = 512, 512
        self.out_path = out_path
        self.model = model
        self.tile_list = tile_list
        self.vrt = vrt
        self.batch_size = 1
        self.add_overlap = add_overlap
        
    def set_tile_mask_dir(self, mask_dir, classes_to_keep=[1], suffix=''):
        self.mask_dir = mask_dir
        self.to_keep = classes_to_keep
        self.suffix = suffix

    def set_batch_size(self, n):
        self.batch_size = n
        
    def write_tiles(self, tqdm, mask=False): 
        tiles = self.tile_list
        
        for tile_i in tqdm(tiles):
            tl_name = os.path.split(tile_i)[1]

            if self.add_overlap:
                buff = self.overlap
                ol_added = True
            else:
                buff = 0
                ol_added = False

            tile = open_tile(tile_i, self.vrt, overlap=buff)
            
            t_image = TiledImageRio(
                tile,
                [self.tiley, self.tilex],
                overlap=self.overlap,
                prep_func=self.model.preprocessing_func,
                ol_added=ol_added
            )

            loader = torch.utils.data.DataLoader(
                t_image,
                batch_size=self.batch_size,
                num_workers=os.cpu_count()-1,
                shuffle=False
            )
            
            image_writer_callback = t_image.get_writer_callback(
                tile_name=tl_name,
                output_dir=self.out_path,
            )
            
            if mask:
                # Assume same tile_name
                mask_name = tl_name.replace('.tif', f'{self.suffix}.tif')
                mask_tile_path = os.path.join(self.mask_dir, mask_name)
                image_writer_callback.set_mask(
                    mask_tile_path,
                    self.to_keep # List
                )

            trainer = pl.Trainer(
                accelerator='gpu',
                callbacks=[image_writer_callback],
                enable_progress_bar=False
                )

            trainer.predict(self.model, loader, return_predictions=False)

class TileSquidger():
    def __init__(self, tile_list, dirs, class_mapping):
        self.set_tiles_to_combine(tile_list)
        self.set_tile_paths(dirs)
        self.set_class_mapping(class_mapping)
        self.set_nodata_vars([])
        self.vectors = []
        self.remap = self.set_remap_for_merge({})
        self.new_map = {}
        
    def set_nodata_vars(self, var_list):
        self.to_drop = var_list

    def set_new_mapping(self, map_dict):
        self.new_map = map_dict
        
    def add_vector_layer(self, vector_src, class_map, col, remove=[]):
        self.vectors.append((vector_src, class_map, col, remove))
    
    def _load_tile(self, path, expand_vars=False, class_name='', drop_vars=[]):
        class_map = self.mapping
        
        src = rio.open_rasterio(
            path,
            masked=False,
            cache=False,
            dtype='int8'
        )

        src.rio.write_nodata(None)
        
        if expand_vars:
            if class_name in class_map.keys():
                names = class_map[class_name]
            else:
                names = dict(zip(classes, [str(c) for c in classes]))
                
            src = self._expand_to_vars(src, names, drop_vars=drop_vars)

        return src
    
    def set_tiles_to_combine(self, tile_list):
        self.tile_names = tile_list
    
    def set_tile_paths(self, paths):
        """Set the directories containing files to combine.
        
        dict{'main':..., 'c':..., ...}
        """
        self.folders = paths
        
    def set_class_mapping(self, class_map):
        """Set the class mapping for output.
        
        dict{'main':..., 'c':..., ...}
        """
        self.mapping = class_map
        
    def set_remap_for_merge(self, mapping):
        self.remap = mapping
        
    def _load_vector_as_raster(self, path, column, rast_like):
        
        bbox = rast_like.rio.bounds()
        
        vdf = gpd.read_file(
            path,
            bbox=bbox,
        )
        
        # Maybe no features in bbox
        if vdf.empty:
            return None
        
        mask_features = make_geocube(
            vector_data=vdf,
            measurements=[column],
            like=rast_like,
            fill=0,
        )
        
        return mask_features[column]
        
    def merge_variables(self, raster_list):
        """This has got rather complicated, refactor"""
        var_list = [list(r.keys()) for r in raster_list]
        
        # Get the duplicates
        seen = set()
        duplicates = set()
        for ds_vars in var_list:
            for v in set(ds_vars):
                if v in seen:
                    duplicates.add(v)
                else:
                    seen.add(v)
        
        non_dups = []
        
        for ds_vars in var_list:
            non_dups.append([x for x in ds_vars if x not in duplicates])
        
        nd = xr.merge(r[non_dups[i]] for i, r in enumerate(raster_list))
        
        for dup in duplicates:
            for i, rast in enumerate(raster_list):
                to_sum = []
                if dup in var_list[i]:
                    to_sum.append(i)
                    
            rasters_wd = [raster_list[j][dup] for j in to_sum]
            nd = nd.assign(sum_vars=sum(rasters_wd))
            nd = nd.rename({'sum_vars': dup})
            
        return nd
    
    def flatten(self, raster_ds):
        
        keys = list(raster_ds.keys())
        map_all = dict(zip(keys, range(1, len(keys)+1)))
        
        data_like = raster_ds[keys[0]]
        arr = np.zeros(data_like.shape)

        for var in list(keys):
            arr += xr.where(raster_ds[var]==1, map_all[var], 0)

        flat = data_like.copy(data=arr)
        flat.name = 'Classes'
        flat = flat.astype('int8')
        
        return flat
    
    def recode_thematic(self, raster, class_map):
        """Return recoded raster.
        
        `class_map` as a dict `from: to`.
        """
        crs = raster.rio.crs
        out_raster = raster.copy()

        for cls in class_map.keys():
            out_raster = xr.where(raster==cls, class_map[cls], out_raster)
            
        
        out_raster.rio.write_crs(crs, inplace=True)
        
        return out_raster
    
    def merge(self, src, rst, rst_remap={}, method=''):
        """The default behaviour is to overwrite, change to `fill`
        to fill zeros only.
        
        `class_maps` should be a list of dicts `from: to` in
        the same order as the `raster_list`.
        """
        if rst_remap:
            rst = self.recode_thematic(rst, rst_remap)
            
        crs = src.rio.crs

        if method=='fill':
            src = xr.where(src==0, rst, src)
        else: # Overwrite!
            src = xr.where(rst!=0, rst, src)
        
        src.rio.write_crs(crs, inplace=True)
        
        return src
    
    def merge_classes(self, raster_list, class_maps=[], onload=True):
        """Add classes from multiple rasters.
        """
        
        # Empty class maps = no remapping
        if not class_maps:
            class_maps = [{} for m in range(len(raster_list))]
        
        if onload:
            img = self._load_tile(raster_list[0])
            
            if class_maps[0]:
                img = self.recode_thematic(img, class_maps[0])
            
            for i, src in enumerate(raster_list[1:]):
                rst = self._load_tile(src)
                
                img = self.merge(img, rst, class_maps[i+1])
                del rst
                       
        else: # list of datasets
            
            img = raster_list[0]
            
            for i, rst in enumerate(raster_list[1:]):
                img = self.merge(img, rst, class_maps[i])
                
        return img
                                    
    def squidge(self, outpath, reduce=[], vector_path=''):
        tiles = self.tile_names
        dirs = self.folders
        maps = self.remap

        for tile in tqdm(tiles):
            rasters = [] 
            ordered_maps = []
            
            for cls in dirs.keys():
                rasters.append(
                    os.path.join(dirs[cls], tile)
                )
                
                if cls in maps:
                    remap = maps[cls]
                else:
                    remap = {}
                
                ordered_maps.append(remap)

            rast_ds = self.merge_classes(rasters, class_maps=ordered_maps, onload=True)

            for vec, class_map, col, ndata in self.vectors:
                vec_ds = self._load_vector_as_raster(vec, col, rast_ds)

                if vec_ds is not None:
                    
                    remap_vec = {}
                    if 'vec' in maps:
                        remap_vec = maps['vec']

                    rast_ds = self.merge(rast_ds, vec_ds, remap_vec)
                    del vec_ds

            if reduce:
                rast_ds = self.reduce_to_proportion(rast_ds, reduce)
            
            rast_ds = rast_ds.transpose('band', 'y', 'x')
    
            outpath_full = os.path.join(outpath, tile)
            
            if vector_path:
                tile_vec = os.path.splitext(tile)[0]
                outpath_vec = os.path.join(vector_path, f'{tile_vec}.shp')
                vectorise_tile(
                    rast_ds,
                    output_path=outpath_vec,
                    mapping=self.new_map
                )

            rast_ds.astype('uint8').rio.to_raster(outpath_full)

            del rast_ds

    def tile_reduce(self, in_path, out_path, res_xy, as_counts=True):
        tiles = self.tile_names

        for t_filename in tqdm(tiles):
            f_tile = rio.open_rasterio(
                os.path.join(in_path, t_filename),
                masked=False,
                cache=False,
            )
            red = expand_to_vars_n_reduce(
                f_tile,
                res_xy,
                names=self.new_map
            )
            outpath_full = os.path.join(out_path, t_filename)
            red.astype('int32').rio.to_raster(outpath_full)

def reduce_to_proportion(img, resolution_xy, as_counts=False, boundary='exact'):
    """Return a proportion RasterDataset (xarray)"""
    x_r, y_r = img.rio.resolution()
    y_r = np.sqrt(y_r**2)
    sx, sy = resolution_xy

    # Cannot sum part pixels
    if (sx%x_r!=0) | (sy%y_r!=0):
        print(f'{resolution_xy} must be a multiple of current resolution')
        
        return
    
    sx = sx/x_r
    sy = sy/y_r
    
    try:
        prop = img.coarsen(x=int(sx), y=int(sy), boundary=boundary).sum()
        prop = prop.rio.write_transform(prop.rio.transform(True))
        
        del img

        if as_counts:
            return prop
        else:
            return prop/(sx*sy)
    
    except ValueError as ve:
        
        print('New resoultion must fit within dimensions with no remainder')

def expand_to_vars_n_reduce(ds, resolution_xy=[], names={}, boundary='exact', as_counts=True):
    # Single band raster
    src = ds.squeeze(drop=True)

    if resolution_xy:
        x_r, y_r = ds.rio.resolution()
        y_r = np.sqrt(y_r**2)
        sx, sy = resolution_xy
        # Cannot sum part pixels
        if (sx%x_r!=0) | (sy%y_r!=0):
            print(f'{resolution_xy} must be a multiple of current resolution')
            return None
        sx = sx/x_r
        sy = sy/y_r
        reduce = True
    else:
        reduce = False

    if not names: # All integer values <= max
        src_max = src.max()
        int_range = np.arange(src_max)+1
        names = dict(zip(int_range, int_range))

    # By band to save memory
    rd_bands = []
    for i, cl in enumerate(names.keys()):
        band = xr.where(src==cl, 1, 0)
        band.name = names[cl]
        if reduce:
            try:
                rd = band.coarsen(x=int(sx), y=int(sy), boundary=boundary).sum()
                rd_bands.append(rd)

            except ValueError as ve:
                print('New resoultion must fit within dimensions with no remainder')
        else:
            rd_bands.append(band)
    
    out = xr.merge(rd_bands)
    # Force update of transform
    out = out.rio.write_transform(out.rio.transform(True))

    if as_counts:
        return out
    else:
        return out/(sx*sy)

def expand_to_vars(ds, names, drop_vars=[]):
        
    src = ds.squeeze(drop=True)
    #classes = range(int(src.max())+1)
    classes = range(np.array(list(names.keys())).max() + 1)

    for cl in classes:
        if cl in names:
            value = xr.where(src==cl, 1, 0)
            value.name = names[cl]
            bands.append(value)

    src = xr.merge(bands)

    del bands

    if drop_vars:
        all_vars = list(src.keys())
        keep = [x for x in all_vars if x not in drop_vars]
        src = src[keep]

    return src

def load_tile(path): 
    
    src = rio.open_rasterio(
        path,
        masked=False,
        cache=False,
        dtype='int8'
    )

    src.rio.write_nodata(None)
    
    return src

def test_reduce(tiles, classes, out_path):

    for fn in tqdm(tiles):
        tile = load_tile(fn)
        tile_x = expand_to_vars(tile, classes)
        reduced = reduce_to_proportion(tile_x, [10, 10], as_counts=True)
        outpath_full = os.path.join(out_path, os.path.split(fn)[1])
        
        reduced.astype('int16').rio.to_raster(outpath_full)

        # Memory leak!

        reduced.close()
        tile.close()
        tile_x.close()

        del reduced
        del tile
        del tile_x

        gc.collect()

def vectorise_tile(raster, dtype='uint8', output_path='', mapping={}):
    # Generator
    vectors = rasterio.features.shapes(
        source=raster.astype(dtype), transform=raster.rio.transform()
    )
    
    values = []
    polys = []
    
    for geom, value in vectors:
        values.append(int(value))
        polys.append(sgeom.shape(geom))

    vec_df = gpd.GeoDataFrame(
        data={'value': values}, geometry=polys, crs=raster.rio.crs
    )
    
    if mapping:
        vec_df['name'] = vec_df['value'].map(mapping)

    if output_path:
        vec_df.to_file(output_path)
    else:
        return vec_df
