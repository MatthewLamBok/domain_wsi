
def part_2_test(data_dict, ai_bool):
    
    slide_id = '52'  
    slide_file_path='/home/mlam/Documents/Research_Project/images_data/IMAGES-Copy/ALL_images/'+slide_id+'.svs'
    h5_file_path_arg = '/home/mlam/Documents/Research_Project/images_data/Output/RESULTS_DIRECTORY_BW_256_v3/'
    h5_file_path =os.path.join(h5_file_path_arg, 'patches', slide_id+'.h5')
    sorted_indices = np.argsort(data_dict[slide_id]['attention_score'], axis=0).flatten()
    sorted_attention_score = data_dict[slide_id]['attention_score'][sorted_indices]
    sorted_coords = data_dict[slide_id]['coords'][sorted_indices]

    #print(sorted_attention_score, sorted_coords)
    
    wsi = openslide.open_slide(slide_file_path)
  
    patches_dataset = Instance_Dataset_heatmap(wsi=wsi,coords=sorted_coords, patch_level= 0, patch_size = 256,slide_id=slide_id)
    print(torch.max(patches_dataset[0][0]),torch.min(patches_dataset[0][0]))
    
    img = patches_dataset[0][0].squeeze().permute(1, 2, 0).numpy()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('image')

    #plt.subplot(1, 2, 2)
    #img = normalize(image_to_plot)
    #plt.imshow(img)
    #plt.title('normalization')
    #plt.axis('off') 
    plt.show()
    
    stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin':       [0.07, 0.99, 0.11],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0],
    }

    # specify stains of input image
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains

    # create stain matrix
    W = np.array([stainColorMap[stain_1],
                stainColorMap[stain_2],
                stainColorMap[stain_3]]).T

    # perform standard color deconvolution
    im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(img*255, W).Stains

    # Display results
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(im_stains[:, :, 0], cmap="gray")
    plt.title(stain_1, fontsize=12)

    plt.subplot(1, 2, 2)
    plt.imshow(im_stains[:, :, 1], cmap="gray")
    _ = plt.title(stain_2, fontsize=12)
    plt.show()
    
        # get nuclei/hematoxylin channel
    im_nuclei_stain = im_stains[:, :, 0]
    if ai_bool:
        from stardist.models import StarDist2D
        StarDist2D.from_pretrained()
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        labels, _ = model.predict_instances(normalize(im_nuclei_stain))

        plt.subplot(1,2,1)
        plt.imshow(normalize(im_nuclei_stain), cmap="gray")
        plt.axis("off")
        plt.title("input image")

        plt.subplot(1,2,2)
        plt.imshow(render_label(labels, img=im_nuclei_stain))
        plt.axis("off")
        plt.title("prediction + input overlay")
        plt.show()
    if not ai_bool:
        

        # segment foreground
        foreground_threshold = 110

        im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
            im_nuclei_stain < foreground_threshold)

        # run adaptive multi-scale LoG filter
        min_radius = 10
        max_radius = 15

        im_log_max, im_sigma_max = htk.filters.shape.cdog(
            im_nuclei_stain, im_fgnd_mask,
            sigma_min=min_radius * np.sqrt(2),
            sigma_max=max_radius * np.sqrt(2),
        )

        # detect and segment nuclei using local maximum clustering
        local_max_search_radius = 10

        im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
            im_log_max, im_fgnd_mask, local_max_search_radius)

        # filter out small objects
        min_nucleus_area = 80

        im_nuclei_seg_mask = htk.segmentation.label.area_open(
            im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

        # compute nuclei properties
        objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

        print ('Number of nuclei = ', len(objProps))

        # Display results
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.imshow(skimage.color.label2rgb(im_nuclei_seg_mask, img, bg_label=0), origin='lower')
        plt.title('Nuclei segmentation mask overlay', fontsize=12)

        plt.subplot(1, 2, 2)
        plt.imshow( img )
        plt.xlim([0, img.shape[1]])
        plt.ylim([0, img.shape[0]])
        plt.title('Nuclei bounding boxes', fontsize=12)

        for i in range(len(objProps)):

            c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
            width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
            height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1

            cur_bbox = {
                'type':        'rectangle',
                'center':      c,
                'width':       width,
                'height':      height,
            }

            plt.plot(c[0], c[1], 'g+')
            mrect = mpatches.Rectangle([c[0] - 0.5 * width, c[1] - 0.5 * height] ,
                                    width, height, fill=False, ec='g', linewidth=2)
            plt.gca().add_patch(mrect)
        plt.show()






def part_3_test(data_dict, display_bool):
    model = StarDist2D.from_pretrained("2D_versatile_he")

    slide_id = '52'  
    slide_file_path='/home/mlam/Documents/Research_Project/images_data/IMAGES-Copy/ALL_images/'+slide_id+'.svs'
    h5_file_path_arg = '/home/mlam/Documents/Research_Project/images_data/Output/RESULTS_DIRECTORY_BW_256_v3/'
    h5_file_path =os.path.join(h5_file_path_arg, 'patches', slide_id+'.h5')
    sorted_indices = np.argsort(data_dict[slide_id]['attention_score'], axis=0).flatten()
    sorted_attention_score = data_dict[slide_id]['attention_score'][sorted_indices]
    sorted_coords = data_dict[slide_id]['coords'][sorted_indices]

    
    
    wsi = openslide.open_slide(slide_file_path)
  
    patches_dataset = Instance_Dataset_heatmap(wsi=wsi,coords=sorted_coords, patch_level= 0, patch_size = 256, slide_id=slide_id)
    print(torch.max(patches_dataset[0][0]),torch.min(patches_dataset[0][0]))
    for i in range(0, 1000):
        crop = patches_dataset[i][0].squeeze().permute(1, 2, 0).numpy()
        if display_bool:
            crop_squid = sq.im.ImageContainer(crop)
            sq.im.segment(
                img=crop_squid,
                layer="image",
                model= model,
                channel=None,
                method=stardist_2D_versatile_he,
                layer_added="segmented_stardist_default",
                prob_thresh=0.3,
                nms_thresh=None,
            )
            print(
                f"Number of segments in crop: {len(np.unique(crop_squid['segmented_stardist_default']))}"
            )
            fig, axes = plt.subplots(1, 2)
            crop_squid.show("image", ax=axes[0])
            _ = axes[0].set_title("H&H")
            crop_squid.show("segmented_stardist_default", cmap="jet", interpolation="none", ax=axes[1])
            _ = axes[1].set_title("segmentation")
            plt.show()
        
        labels = stardist_2D_versatile_he(crop, model= model, nms_thresh=None, prob_thresh=0.3)
        props = regionprops(labels)
        cell_sizes = [prop.area for prop in props]  
        num_cells = len(props) 
        centroids = np.array([prop.centroid for prop in props])  
        distances = cdist(centroids, centroids)
        cell_perimeters = [prop.perimeter for prop in props]
        cell_circularity = [4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0 for prop in props]
        cell_info = {
            'num_cells': num_cells,
            'cell_sizes': cell_sizes,
            'cell_perimeters': cell_perimeters,
            'cell_circularity': cell_circularity,
            'distances': distances,
        }
        print(cell_info['num_cells'])
        exit()