import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# Função histograma
def histograma(img):
    vet = np.zeros((256), dtype=np.uint16)
    img = np.uint8(img)
    for l in range(0, altura):
        for c in range(0, largura):
            vet[img[l,c]] += 1
    plt.plot(vet)
    plt.title('Histograma')
    plt.ylabel('Quantidade de pixels')
    plt.xlabel('Intensidade de cor')
    string = str(int(time.time()))
    print(string)
    plt.savefig('processadas/'+nome+'_histograma_'+string)
    axes = plt.gca()
    axes.set_ylim(0,)
    plt.show()

# Altera a intensidade do pixel, para mais ou para menos
def alteraBrilho(img, intensidade):
    nova = np.zeros((altura,largura))
    for l in range(0, altura):
        for c in range(0, largura):
            nova[l,c] = max(min(img[l,c]+intensidade,255),0)
    cv2.imshow('Imagem alterada', np.uint8(nova))
    cv2.imwrite('processadas/'+nome+'_altera_brilho_'+str(intensidade)+'.png', np.uint8(nova))
    cv2.waitKey(0)
    return np.uint8(nova)

# Aplica o filtro de média à imagem
def media(img):
    nova = np.zeros((altura,largura))
    for l in range(0, altura):
        for c in range(0, largura):
            soma = 0
            interacoes = 0
            for lm in range (-1,2):
                for cm in range (-1,2):
                    if(l+lm>=0 and c+cm>=0 and l+lm<altura and c+cm<largura):
                        soma += img[l+lm,c+cm]
                        interacoes+=1
            nova[l,c] = soma/interacoes
    cv2.imshow('Imagem - Filtro de média', np.uint8(nova))
    cv2.imwrite('processadas/' + nome + '_media.png', np.uint8(nova))
    cv2.waitKey(0)
    return np.uint8(nova)

# Aplica o filtro de mediana à imagem
def mediana(img):
    nova = np.zeros((altura,largura))
    m = np.zeros((9), dtype=np.int16)

    for l in range(0, altura):
        for c in range(0, largura):
            i = 0
            for lm in range (-1,2):
                for cm in range (-1,2):
                    if (l + lm >= 0 and c + cm >= 0 and l + lm < altura and c + cm < largura):
                        m[i] = img[l+lm,c+cm]
                    else:
                        m[i] = 0
                    i += 1
            nova[l,c] = np.median(m)
    cv2.imshow('Imagem - Filtro de média', np.uint8(nova))
    cv2.imwrite('processadas/' + nome + '_mediana.png', np.uint8(nova))
    cv2.waitKey(0)
    return np.uint8(nova)

# Aplica filtros na imagem com o uso de máscaras
def mascara(img, tipo):
    nova = np.zeros((altura,largura))
    m = np.zeros((3, 3))

    if(tipo=="sobel"):
        m[0,0] = 1
        m[0, 1] = 0
        m[0, 2] = -1
        m[1, 0] = 2
        m[1, 1] = 0
        m[1, 2] = -2
        m[2, 0] = 1
        m[2, 1] = 0
        m[2, 2] = -1
    elif (tipo == "prewitt"):
        m[0, 0] = -1
        m[0, 1] = 0
        m[0, 2] = 1
        m[1, 0] = -1
        m[1, 1] = 0
        m[1, 2] = 1
        m[2, 0] = -1
        m[2, 1] = 0
        m[2, 2] = 1
    elif (tipo == "laplaciano"):
        m[0, 0] = 0
        m[0, 1] = -1
        m[0, 2] = 0
        m[1, 0] = -1
        m[1, 1] = 4
        m[1, 2] = -1
        m[2, 0] = 0
        m[2, 1] = -1
        m[2, 2] = 0
    elif (tipo == "passa-alta"):
        m[0, 0] = -1
        m[0, 1] = -1
        m[0, 2] = -1
        m[1, 0] = -1
        m[1, 1] = 8
        m[1, 2] = -1
        m[2, 0] = -1
        m[2, 1] = -1
        m[2, 2] = -1

    for l in range(0, altura):
        for c in range(0, largura):
            soma = 0
            for lm in range (-1,2):
                for cm in range (-1,2):
                    if (l + lm >= 0 and c + cm >= 0 and l + lm < altura and c + cm < largura):
                        soma = soma + (int(m[lm+1,cm+1]) * img[l+lm,c+cm])
                    else:
                        soma = soma + (int(m[lm+1,cm+1]) * img[l,c])
            nova[l,c] = max(min(soma, 255), 0)
    cv2.imshow('Imagem - Filtro mask', np.uint8(nova))
    cv2.imwrite('processadas/' + nome + '_mascara_'+tipo+'.png', nova)
    cv2.waitKey(0)
    return np.uint8(nova)

def linhas(img, tipo):
    nova = np.zeros((altura,largura))
    totalh = processaMascara("horizontal",nova,img)
    totalv = processaMascara("vertical", nova, img)
    total45 = processaMascara("45", nova, img)
    totalm45 = processaMascara("-45", nova, img)

    if(totalh>totalv and totalh>total45 and totalh>totalm45):
        print("Horizontal")
    elif(totalv>totalh and totalv>total45 and totalv>totalm45):
        print("Vertical")
    elif(total45 > totalh and total45 > totalv and total45 > totalm45):
        print("Inclinada (45 graus)")
    elif(totalm45 > totalh and totalm45 > total45 and totalm45 > totalv):
        print("Inclinada (-45 graus)")

def processaMascara(tipo,nova,img):
    m = np.zeros((3, 3))
    if (tipo == "horizontal"):
        m[0, 0] = -1
        m[0, 1] = -1
        m[0, 2] = -1
        m[1, 0] = 2
        m[1, 1] = 2
        m[1, 2] = 2
        m[2, 0] = -1
        m[2, 1] = -1
        m[2, 2] = -1
    elif (tipo == "vertical"):
        m[0, 0] = -1
        m[0, 1] = 2
        m[0, 2] = -1
        m[1, 0] = -1
        m[1, 1] = 2
        m[1, 2] = -1
        m[2, 0] = -1
        m[2, 1] = 2
        m[2, 2] = -1
    elif (tipo == "45"):
        m[0, 0] = -1
        m[0, 1] = -1
        m[0, 2] = 2
        m[1, 0] = -1
        m[1, 1] = 2
        m[1, 2] = -1
        m[2, 0] = 2
        m[2, 1] = -1
        m[2, 2] = -1
    elif (tipo == "-45"):
        m[0, 0] = 2
        m[0, 1] = -1
        m[0, 2] = -1
        m[1, 0] = -1
        m[1, 1] = 2
        m[1, 2] = -1
        m[2, 0] = -1
        m[2, 1] = -1
        m[2, 2] = 2

    cont = 0
    for l in range(0, altura):
        for c in range(0, largura):
            soma = 0
            for lm in range(-1, 2):
                for cm in range(-1, 2):
                    if (l + lm >= 0 and c + cm >= 0 and l + lm < altura and c + cm < largura):
                        soma = soma + (int(m[lm + 1, cm + 1]) * img[l + lm, c + cm])
                    else:
                        soma = soma + (int(m[lm + 1, cm + 1]) * img[l, c])
            valor = max(min(soma, 255), 0)
            #nova[l, c] = valor

            if (valor > 250):
                cont += 1
    return cont

def encontrarsemente(img):
    pyinicial = -1
    pxinicial = -1
    for l in range(0, altura):
        for c in range(0, largura):
            if (img[l, c] < 50 and pxinicial < 0 and pyinicial < 0):
                pxinicial = l
                pyinicial = c
    return pxinicial,pyinicial

def recursiva(matriz):
    py, px = encontrarsemente(matriz)
    if (py < 0 and px < 0):
        return 0;
    else:
        for l in range(py, altura):
            for c in range(px, largura):
                if (matriz[l, c] < 50):
                    matriz[l, c] = 255
                elif (matriz[l, c] > 50):
                    break
    return 1 + (recursiva(matriz))

# Aplica o filtro de média à imagem
def crescimento(img):
    nova = np.zeros((altura,largura))
    nova = img

    print("Total de objetos encontrados: ", recursiva(nova))



#Definições de arquivo
nome = 'objetos-diferentes'
imagem = cv2.imread('imagens/'+nome+'.jpg')
cv2.imwrite('processadas/' + nome + '_original.png', imagem)
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imwrite('processadas/' + nome + '_cinza.png', imagem)
#cv2.imshow('Imagem original', imagem)
#cv2.waitKey(0)

#Informações de tamanho da imagem
altura = imagem.shape[0]
largura = imagem.shape[1]

crescimento(imagem)


#histograma(imagem)
#histograma(mascara(imagem,"passa-alta"))
#histograma(mascara(imagem,"laplaciano"))
#histograma(mascara(imagem,"sobel"))

#alteraBrilho(imagem,-90)
#alteraBrilho(imagem,90)
#media(imagem)
#mediana(imagem)
#mascara(imagem,"sobel")
