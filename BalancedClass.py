def make_weights_for_balanced_classes(images, sub_images, nclasses):  
    
    count = [0] * nclasses                                                       
    for item in sub_images: 
        count[images[item]] += 1  
                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    
    weight = [.0] * len(sub_images)
    for idx, item in enumerate(sub_images):
        weight[idx] = weight_per_class[images[item]]
                                 
    return weight  