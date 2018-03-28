import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogramaOriginal(img):
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist_full)
    plt.xlim([0,256])
    plt.show()


# Função histograma
def histograma(img):
    img = img.copy()
    vet = np.zeros((256), dtype=np.int16)
    for l in range(0, altura):
        for c in range(0, largura):
            vet[img[l,c]] += 1

    plt.plot(vet)
    plt.title('Histograma')
    plt.ylabel('Quantidade de pixels')
    plt.xlabel('Intensidade de cor')
    plt.savefig('processadas/'+nome+'_histograma')
    axes = plt.gca()
    #axes.set_xlim([0,255])
    axes.set_ylim(0,)
    plt.show()

# Altera a intensidade do pixel, para mais ou para menos
def alteraBrilho(img, intensidade):
    img = img.copy()
    for l in range(0, altura):
        for c in range(0, largura):
            img[l,c] = max(min(img[l,c]+intensidade,255),0)
    cv2.imshow('Imagem alterada', img)
    cv2.imwrite('processadas/'+nome+'_altera_brilho_'+str(intensidade)+'.png', img)
    cv2.waitKey(0)


def media(img):
    img = img.copy()
    for l in range(0, altura):
        for c in range(0, largura):
            soma = 0
            interacoes = 0
            for lm in range (-1,2):
                for cm in range (-1,2):
                    if(l+lm>=0 and c+cm>=0 and l+lm<altura and c+cm<largura):
                        soma += img[l+lm,c+cm]
                        interacoes+=1
            img[l,c] = soma/interacoes
    cv2.imshow('Imagem - Filtro de média', img)
    cv2.imwrite('processadas/' + nome + '_media.png', img)
    cv2.waitKey(0)

def mediana(img):
    img = img.copy()
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
            img[l,c] = np.median(m)
    cv2.imshow('Imagem - Filtro de média', img)
    cv2.imwrite('processadas/' + nome + '_mediana.png', img)
    cv2.waitKey(0)


def mascara(img, tipo):
    img = img.copy()
    m = np.zeros((3,3), dtype=np.int16)
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

    for l in range(0, altura):
        for c in range(0, largura):
            soma = 0
            #soma = (img[l - 1, c - 1] * 1 + img[l - 1, c] * 2 + img[l - 1, c + 1] * 1
            #       + img[l, c - 1] * 0 + img[l, c] * 0 + img[l, c + 1] * 0
            #       + img[l + 1, c - 1] * -1 + img[l + 1, c] * -2 + img[l + 1, c + 1] * -1)

            for lm in range (-1,2):
                for cm in range (-1,2):
                    if (l + lm >= 0 and c + cm >= 0 and l + lm < altura and c + cm < largura):
                       soma += img[l+lm,c+cm] * m[lm+1,cm+1]
            img[l,c] = max(min(soma,255),0)
    cv2.imshow('Imagem - Filtro mask', img)
    cv2.imwrite('processadas/' + nome + '_mascara_'+tipo+'.png', img)
    cv2.waitKey(0)

#Definições de arquivo
nome = 'monica'

imagem = cv2.imread('imagens/'+nome+'.jpg')
cv2.imwrite('processadas/' + nome + '_original.png', imagem)
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imwrite('processadas/' + nome + '_cinza.png', imagem)

#Informações de tamanho da imagem
altura = imagem.shape[0]
largura = imagem.shape[1]

cv2.imshow('Imagem original', imagem)
cv2.waitKey(0)


#alteraBrilho(imagem,70)
#histogramaOriginal(imagem)
#histograma(imagem)
#alteraBrilho(imagem,-90)

#mediana(imagem)
mascara(imagem,"laplaciano")
#media(imagem)
#mediana(imagem)
