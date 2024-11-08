
import rioxarray as rio
import os
import xarray as xr
from geocube.api.core import make_geocube
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import utils
import pickle
import rasterio
from tqdm.notebook import tqdm
import glob

MD_PATHS = utils.MD_PATHS
    
def get_class_mapping(paths, key='dict_new_names'):
    mapping = {}
    for cl in paths.keys():
        with open(paths[cl], 'rb') as f:
            mapping[cl] = pickle.load(f)[key]
    return mapping

def get_new_merged_class_mapping():
    rast_maps = get_class_mapping(MD_PATHS)
    
    vec_class_map = {
        0: 'NO CLASS',
        1: 'River',
        2: 'Lake',
        3: 'Pond',
        4: 'Suburban',
        5: 'Cemeteries',
        6: 'Playing fields (i.e. grass without trees)',
        7: 'Railway verges',
        8:  'Railway',
        9: 'Road',
        10: 'Road verges',
        11: 'Dense urban'
    }
        
    all_classes = rast_maps['all'].copy()
    new_ints = [x+len(rast_maps['all']) for x in vec_class_map.keys()]

    all_classes.update(dict(zip(new_ints, vec_class_map.values())))

    # Different nodata class
    del all_classes[14]

    all_classes_rv = dict((v,k) for k,v in all_classes.items())
    
    remap_merge = {}
    rast_maps.pop('all', None)
    rast_maps.pop('main', None)
    rast_maps.update({'vec': vec_class_map})

    for m in rast_maps.keys():
        nm = {}
        for k in rast_maps[m].keys():
            v = rast_maps[m][k]
            nm[k] = all_classes_rv[v]
        remap_merge[m] = nm

    return remap_merge

def test(n=1):
    paths = glob.glob('/home/jovyan/work/outputs/nlees_com_main_v2/*.tif')
    files = [os.path.split(p)[1] for p in paths]
    
    ROOT = '/home/jovyan/work/outputs'
    dirs = {
        'c': os.path.join(ROOT, 'nlees_com_c_v2'),
        'd': os.path.join(ROOT, 'nlees_com_d_v2'),
        'e': os.path.join(ROOT, 'nlees_com_e_v2'),
    }
    mapping = get_class_mapping(MD_PATHS)
    
    reducer = TileSquidger(files[:n], dirs, mapping)
    reducer.set_nodata_vars(['NO CLASS'])

    vec_layer_path = '/home/jovyan/work/project_data/os_ndg/nlees/os_ngd_recode.shp'
    vec_class_map = {
        0: 'NA',
        1: 'River',
        2: 'Pond',
        3: 'Suburban',
        4: 'Railway',
        5: 'Railway verges',
        6: 'Road',
        7: 'Road verges',
        8: 'Dense urban',
    }
    
    reducer.add_vector_layer(vec_layer_path, vec_class_map, 'class', remove=[])
    reducer.set_remap_for_merge(get_new_merged_class_mapping())
    
    reducer.squidge('/home/jovyan/work/outputs/flatten')

class TileSquidger():
    def __init__(self, tile_list, dirs, class_mapping):
        self.set_tiles_to_combine(tile_list)
        self.set_tile_paths(dirs)
        self.set_class_mapping(class_mapping)
        self.set_nodata_vars([])
        self.vectors = []
        self.remap = self.set_remap_for_merge({})
        
    def set_nodata_vars(self, var_list):
        self.to_drop = var_list
        
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
    
    def _expand_to_vars(self, ds, names, drop_vars=[]):
        
        src = ds.squeeze(drop=True)
        bands = []
        classes = range(int(src.max())+1)

        for cl in classes:
            value = xr.where(src==cl, 1, 0)
            value.name = names[cl]
            bands.append(value)

        src = xr.merge(bands)
        if drop_vars:
            all_vars = list(src.keys())
            keep = [x for x in all_vars if x not in drop_vars]
            src = src[keep]
            
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

        for cls in class_map.keys():
            raster = xr.where(raster==cls, class_map[cls], raster)
        
        raster.rio.write_crs(crs, inplace=True)
        
        return raster
    
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
        else:
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
                img = self.merge(img, rst, class_maps[i])
                       
        else: # list of datasets
            
            img = raster_list[0]
            
            for i, rst in enumerate(raster_list[1:]):
                img = self.merge(img, rst, class_maps[i])
                
        return img
                                    
    def squidge(self, outpath, reduce=[]):
        tiles = self.tile_names
        dirs = self.folders

        for tile in tiles:
            rasters = []
            maps = self.remap
            
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

            if reduce:
                rast_ds = self.reduce_to_proportion(rast_ds, reduce)
            
            rast_ds = rast_ds.transpose('band', 'y', 'x')
    
            outpath_full = os.path.join(outpath, tile)
            rast_ds.astype('uint8').rio.to_raster(outpath_full)

            del rast_ds
            #return rast_ds
        
        
                
            
        
