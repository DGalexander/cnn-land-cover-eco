## Perform prediction on full tiles using a trained model on a test set

import os, pickle
import land_cover_analysis as lca
import land_cover_models as lcm
import loadpaths 

path_dict = loadpaths.loadpaths()

def predict_segmentation_network(filename_model=None, 
                                 folder_model=path_dict['models'],
                                 padding=44, 
                                 dissolve_small_pols=True,
                                dissolve_threshold=1000,  # only used if dissolve_small_pols=True AND use_class_dependent_area_thresholds=False
                                use_class_dependent_area_thresholds=False,
                                file_path_class_dependent_area_thresholds=None,
                                clip_to_main_class=False,
                                col_name_class=None,  # name of column in main predictions shapefile that contains the class label (if None, found automatically if only one candidate column exists)
                                main_class_clip_label='D',
                                skip_factor=16,
                                save_shp_prediction=True,
                                parent_save_folder=path_dict['save_folder'],
                                override_with_fgh_layer=False,
                                subsample_tiles_for_testing=False,
                                dir_im_pred=path_dict['im_tiles'],
                                tifs_in_subdirs=False,
                                dir_mask_eval=None,
                                fgh_layer_path='../content/evaluation_polygons/landscape_character_2022_FGH-override/landscape_character_2022_FGH-override.shp',
                                mask_suffix='_lc_2022_main_mask.tif',
                                parent_dir_tile_mainpred=path_dict['parent_dir_tile_mainpred'],
                                tile_outlines_shp_path='../content/evaluation_sample_50tiles/eval_all_tile_outlines/eval_all_tile_outlines.shp',  # used BOTH for selecting tiles to predict AND for clipping predictions to tile outlines
                                use_tile_outlines_shp_to_predict_those_tiles_only=False,
                                delete_individual_tile_predictions=False,
                                merge_tiles_into_one_shp=True,
                                ):
    '''
    parent_save_folder: str, path to folder where to save the results
    dir_im_pred: str, path to folder with full images tiles to predict
    dir_mask_eval: str, path to folder with mask tiles to evaluate predictions. If None, no evaluation is performed
    fgh_layer_path: str, path to shapefile with manual FGH layer to override main-class predictions with
    mask_suffix: str, suffix of mask files in dir_mask_eval
    parent_dir_tile_mainpred: str, path to folder with main class predictions of full tiles
    tile_outlines_shp_path: str, path to shapefile with tile outlines to predict. If use_tile_outlines_shp_to_predict_those_tiles_only is True, only these tiles are predicted

    '''
    lca.check_torch_ready(check_gpu=True, assert_versions=True)
    assert (use_tile_outlines_shp_to_predict_those_tiles_only is False) or (override_with_fgh_layer is False), 'Currently not implemented to only use selection of tiles for FGH override (as is done in tile prediction wrapper). Either implement or remove this assert and always override with FGH layer on all tiles'
    datapath_model = os.path.join(folder_model, filename_model)
    assert os.path.exists(datapath_model), f'Model file not found at {datapath_model}. Have you set the correct models path in content/data_paths.json?'
    assert os.path.exists(dir_im_pred), f'Image tiles folder not found at {dir_im_pred}. (Have you set the correct image tiles path in content/data_paths.json?)'
    ## Load model (ensure folder is correct):
    LCU = lcm.load_model_auto(filename=filename_model, folder=folder_model)
    LCU.eval() 

    if use_class_dependent_area_thresholds:
        dissolve_threshold = None  # reset if using dict of class-dependent thresholds
        if file_path_class_dependent_area_thresholds is None:
            file_path_class_dependent_area_thresholds = os.path.join(path_dict['repo'], 'content/area_threshold_combinations/th-combi-2023-08-22.json')
            print(f'Using default class-dependent area thresholds from {file_path_class_dependent_area_thresholds}')
        class_dependent_area_thresholds, dissolve_threshold = lca.load_area_threshold_json(file_path_class_dependent_area_thresholds)
        if dissolve_threshold is None:  # not defined in json file:
            dissolve_threshold = 0
            print('WARNING: dissolve_threshold not defined in json file, using 0 instead')
        name_combi = file_path_class_dependent_area_thresholds.split('/')[-1].split('.')[0]
        print(f'Using class-dependent area thresholds from {name_combi}')
    else:
        name_combi = None
        class_dependent_area_thresholds = None

    if dissolve_small_pols and (not use_class_dependent_area_thresholds):
        dissolved_name = '_dissolved' + str(dissolve_threshold) + 'm2'
    elif dissolve_small_pols and use_class_dependent_area_thresholds:
        dissolved_name = f'_dissolved-{name_combi}'
    else:
        dissolved_name = '_notdissolved'
    if clip_to_main_class:
        dissolved_name = dissolved_name + f'_clipped{main_class_clip_label}'
    identifier = 'predictions_' + LCU.model_name + dissolved_name + f'_padding{padding}'
    save_folder = os.path.join(parent_save_folder, identifier, 'individual_tiles')

    if use_tile_outlines_shp_to_predict_those_tiles_only:
        df_tile_outlines = lca.load_pols(tile_outlines_shp_path)
        tile_name_col = 'PLAN_NO'
        list_tile_names_to_predict = df_tile_outlines[tile_name_col].unique().tolist()
        print(f'Predicting only the following number of tiles: {len(list_tile_names_to_predict)}')
    else:
        list_tile_names_to_predict = None

    ## Predict full tiles of test set:
    tmp_results = lcm.tile_prediction_wrapper(model=LCU, save_shp=save_shp_prediction,
                                dir_im=dir_im_pred, list_tile_names_to_predict=list_tile_names_to_predict,
                                dir_mask_eval=dir_mask_eval,
                                save_folder=save_folder, dissolve_small_pols=dissolve_small_pols, 
                                area_threshold=dissolve_threshold, 
                                use_class_dependent_area_thresholds=use_class_dependent_area_thresholds,
                                class_dependent_area_thresholds=class_dependent_area_thresholds,
                                name_combi_area_thresholds=name_combi,
                                skip_factor=skip_factor, tifs_in_subdirs=tifs_in_subdirs,
                                padding=padding, mask_suffix=mask_suffix,
                                clip_to_main_class=clip_to_main_class, main_class_clip_label=main_class_clip_label, 
                                col_name_class=col_name_class,
                                parent_dir_tile_mainpred=parent_dir_tile_mainpred, tile_outlines_shp_path=tile_outlines_shp_path,
                                subsample_tiles_for_testing=subsample_tiles_for_testing)

    ## Save results as pickle:
    with open(os.path.join(save_folder, 'summary_results.pkl'), 'wb') as f:
        pickle.dump(tmp_results, f)
    print('\nResults saved!\n\n')

    if merge_tiles_into_one_shp:
        ## Merge all tiles into one shapefile:
        lca.merge_individual_shp_files(dir_indiv_tile_shp=save_folder,
                                        delete_individual_shp_files=False)  # set to False because they are needed for FGH override

    ## Override predictions with manual FGH layer:
    if override_with_fgh_layer:
        assert clip_to_main_class is False, 'Expected that FGH override would only happen on main class predictions, but clip_to_main_class is set to True which indicates that these are detailed class predictions'
        print('######\n\nOverride predictions with manual FGH layer\n\n######')
        save_folder_fgh = lca.override_predictions_with_manual_layer(filepath_manual_layer=fgh_layer_path,  #'/home/tplas/data/gis/tmp_fgh_layer/tmp_fgh_layer.shp', 
                                                                tile_predictions_folder=save_folder, 
                                                                new_tile_predictions_override_folder=None, verbose=1)

        if merge_tiles_into_one_shp:
            ## Merge all FGH_override tiles into one shapefile:
            lca.merge_individual_shp_files(dir_indiv_tile_shp=save_folder_fgh, 
                                        delete_individual_shp_files=delete_individual_tile_predictions)

if __name__ == '__main__':

    dict_cnns_best = {  ## determined in `Evaluate trained network.ipynb`
        'main': 'main_LCU_2023-04-24-1259.pth',
        'C': 'C_LCU_2023-04-21-1335.pth',
        'D': 'D_LCU_2023-04-25-2057.pth',
        'E': 'E_LCU_2023-04-24-1216.pth'
    }

    for model_use in ['main']:
        predict_segmentation_network(filename_model=dict_cnns_best[model_use], 
                                    clip_to_main_class=False if model_use == 'main' else True, 
                                    col_name_class='lc_label',
                                    main_class_clip_label=model_use,
                                    dissolve_small_pols=False,
                                    dissolve_threshold=10, 
                                    use_class_dependent_area_thresholds=False,
                                    # override_with_fgh_layer=True if model_use == 'main' else False,
                                    override_with_fgh_layer=False,
                                    dir_im_pred='/home/david/Documents/ADP/test/',
                                    parent_dir_tile_mainpred='/home/david/predictions_gis/all_pd_tiles_notdissolved/',
                                    subsample_tiles_for_testing=False,
                                    tile_outlines_shp_path='../content/tiles_qr/tiles_qr.shp',
                                    use_tile_outlines_shp_to_predict_those_tiles_only=False,
                                    delete_individual_tile_predictions=False,         
                                    parent_save_folder='/home/david/predictions_gis/all_tiles_pd_notdissolved/',
                                    merge_tiles_into_one_shp=False                      
                                    )