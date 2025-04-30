# network_grouping.py

network_map = {
    "Visual": [b'7Networks_LH_Vis_1', b'7Networks_LH_Vis_2', b'7Networks_LH_Vis_3', b'7Networks_LH_Vis_4', 
               b'7Networks_LH_Vis_5', b'7Networks_LH_Vis_6', b'7Networks_LH_Vis_7', b'7Networks_LH_Vis_8',
               b'7Networks_LH_Vis_9', b'7Networks_RH_Vis_1', b'7Networks_RH_Vis_2', b'7Networks_RH_Vis_3',
               b'7Networks_RH_Vis_4', b'7Networks_RH_Vis_5', b'7Networks_RH_Vis_6', b'7Networks_RH_Vis_7',
               b'7Networks_RH_Vis_8'],
    "Somatomotor": [b'7Networks_LH_SomMot_1', b'7Networks_LH_SomMot_2', b'7Networks_LH_SomMot_3', 
                    b'7Networks_LH_SomMot_4', b'7Networks_LH_SomMot_5', b'7Networks_LH_SomMot_6', 
                    b'7Networks_RH_SomMot_1', b'7Networks_RH_SomMot_2', b'7Networks_RH_SomMot_3',
                    b'7Networks_RH_SomMot_4', b'7Networks_RH_SomMot_5', b'7Networks_RH_SomMot_6'],
    "Dorsal Attention": [b'7Networks_LH_DorsAttn_Post_1', b'7Networks_LH_DorsAttn_Post_2', 
                         b'7Networks_LH_DorsAttn_Post_3', b'7Networks_LH_DorsAttn_Post_4', 
                         b'7Networks_LH_DorsAttn_Post_5', b'7Networks_LH_DorsAttn_Post_6',
                         b'7Networks_LH_DorsAttn_PrCv_1', b'7Networks_RH_DorsAttn_Post_1', 
                         b'7Networks_RH_DorsAttn_Post_2', b'7Networks_RH_DorsAttn_Post_3',
                         b'7Networks_RH_DorsAttn_Post_4', b'7Networks_RH_DorsAttn_Post_5',
                         b'7Networks_RH_DorsAttn_PrCv_1'],
    "Salience and Ventral Attention": [b'7Networks_LH_SalVentAttn_ParOper_1', 
                                       b'7Networks_LH_SalVentAttn_FrOperIns_1', 
                                       b'7Networks_RH_SalVentAttn_TempOccPar_1', 
                                       b'7Networks_RH_SalVentAttn_TempOccPar_2', 
                                       b'7Networks_RH_SalVentAttn_FrOperIns_1'],
    "Limbic": [b'7Networks_LH_Limbic_OFC_1', b'7Networks_LH_Limbic_TempPole_1', 
               b'7Networks_LH_Limbic_TempPole_2', b'7Networks_RH_Limbic_OFC_1', 
               b'7Networks_RH_Limbic_TempPole_1'],
    "Control": [b'7Networks_LH_Cont_Par_1', b'7Networks_LH_Cont_PFCl_1', 
                b'7Networks_LH_Cont_pCun_1', b'7Networks_LH_Cont_Cing_1', 
                b'7Networks_RH_Cont_Par_1', b'7Networks_RH_Cont_Par_2', 
                b'7Networks_RH_Cont_PFCl_1', b'7Networks_RH_Cont_PFCl_2', 
                b'7Networks_RH_Cont_PFCl_3', b'7Networks_RH_Cont_PFCl_4'],
    "Default Mode": [b'7Networks_LH_Default_Temp_1', b'7Networks_LH_Default_Temp_2', 
                     b'7Networks_LH_Default_Par_1', b'7Networks_LH_Default_Par_2', 
                     b'7Networks_LH_Default_PFC_1', b'7Networks_LH_Default_PFC_2', 
                     b'7Networks_LH_Default_PFC_3', b'7Networks_LH_Default_PFC_4', 
                     b'7Networks_LH_Default_PFC_5', b'7Networks_LH_Default_PFC_6', 
                     b'7Networks_LH_Default_PFC_7', b'7Networks_LH_Default_pCunPCC_1',
                     b'7Networks_LH_Default_pCunPCC_2', b'7Networks_RH_Default_Par_1', 
                     b'7Networks_RH_Default_Temp_1', b'7Networks_RH_Default_Temp_2', 
                     b'7Networks_RH_Default_Temp_3', b'7Networks_RH_Default_PFCv_1', 
                     b'7Networks_RH_Default_PFCv_2', b'7Networks_RH_Default_PFCdPFCm_1',
                     b'7Networks_RH_Default_PFCdPFCm_2', b'7Networks_RH_Default_PFCdPFCm_3', 
                     b'7Networks_RH_Default_pCunPCC_1', b'7Networks_RH_Default_pCunPCC_2']
}

# map regions to the above defined networks
def map_regions_to_networks(labels):
    network_groupings = {network: [] for network in network_map}

    for idx, label in enumerate(labels):
        for network, regions in network_map.items():
            if label in regions:
                network_groupings[network].append(idx)
    
    return network_groupings