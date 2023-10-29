import numpy as np

'''
Функция проходится по всем пикселям изображения,
если находит слабый пиксель, проверяет если ли соседний сильный пиксель,
если есть, слабый пиксель становится сильным.
'''
def DFS(img) :
    for i in range(1, int(img.shape[0] - 1)) :
        for j in range(1, int(img.shape[1] - 1)) :
            if(img[i, j] == 1) :
                t_max = max(img[i-1, j-1], img[i-1, j], img[i-1, j+1], img[i, j-1],
                            img[i, j+1], img[i+1, j-1], img[i+1, j], img[i+1, j+1])
                if(t_max == 2) :
                    img[i, j] = 2
                
                    
# Threshold
def thresholding(img) :
    low_ratio = 0.10
    high_ratio = 0.30
    diff = np.max(img) - np.min(img)
    t_low = np.min(img) + low_ratio * diff
    t_high = np.min(img) + high_ratio * diff
    
    temp_img = np.copy(img)

    for i in range(1, int(img.shape[0] - 1)) :
        for j in range(1, int(img.shape[1] - 1)) :
            # Сильные пиксели
            if(img[i, j] > t_high) :
                temp_img[i, j] = 2
            # Слабые пиксели
            elif(img[i, j] < t_low) :
                temp_img[i, j] = 0
            # Пиксели попавшие в промежуток
            else :
                temp_img[i, j] = 1
    
    #Всклячаем слабые пиксели, которые находятся рядом с сильными пикселями
    total_strong = np.sum(temp_img == 2)
    while(1) :
        DFS(temp_img)
        # Применяем функцию DFS, пока изображение не перестанет изменяться
        if(total_strong == np.sum(temp_img == 2)) :
            break
        total_strong = np.sum(temp_img == 2)
    
    # Удаляем оставшиеся слабые пиксели
    for i in range(1, int(temp_img.shape[0] - 1)) :
        for j in range(1, int(temp_img.shape[1] - 1)) :
            if(temp_img[i, j] == 1) :
                temp_img[i, j] = 0
    
    temp_img = temp_img/np.max(temp_img)
    return temp_img    