import time

def make_weights_for_balanced_classes(images, sub_images, nclasses):  
    
    start_time = time.time()
    print("Calculando repeticiones por clase")                      
    count = [0] * nclasses                                                       
    for item in sub_images: 
        count[images[item]] += 1  
    print(f'Repeticiones calculadas, tiempo: {time.time() - start_time}')
                     
    print("Calculando los pesos por clase")                                      
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    
    weight = [.0] * len(sub_images)
    start_time = time.time()
    print("Pasando los pesos para cada imagen")
    for idx, item in enumerate(sub_images):
        weight[idx] = weight_per_class[images[item]]
          
    print(f"Pesos pasados, tiempo: {time.time() - start_time}")                                       
                              
    return weight  