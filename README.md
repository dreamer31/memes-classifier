# Meme Classifier

Herramienta para clasificar imágenes entre memes, no memes y stickers además de filtrar memes por tópicos. Desarrollada utilizando redes neuronales.

## Instalación

1. Descargar repositorio.
2. Descargar peso del modelo a utilizar.
3. Instalar requeriments  ```pip install -r requeriments.txt```
4. Poner las imágenes a clasificar en la carpeta img_class.
5. Ejecutar archivo run.py con los parámetros correspondientes.

## Modo de uso

```bash
python run.py <model_name> <mode_classifier> <move_image> <show_info>
```

| Parametro       | Descripción                                                         | Valores                                                               |
|-----------------|---------------------------------------------------------------------|-----------------------------------------------------------------------|
| model_name      | Indica el modelo que se utilizara para clasificar                   | "bert"                                                                |
| mode_classifier | Indica que tipo de clasificación que quiere realizar                | "classify"                                                            |
| move_image      | Indica si se quiere mover las imágenes a su carpeta correspondiente | True: Mueve las imágenes False: No las mueve                          |
| show_info       | Indica si se quiere mostrar feedback de la clasificación            | True: Muestra feedback de clasificación False: No se muestra feedback |


## Ejemplo de uso

A continuación se presenta un ejemplo de uso con el modelo de Bert

```Python
python run.py bert classify True True
```

En este caso las imágenes serán movidas a las carpetas correspondientes, mostrando el feedback en consola.

## Pesos

Bert CNN Classificator: https://drive.google.com/file/d/1-MJUZEOy3wNIQD0E9bDK7FVsZ7823JW9/view?usp=sharing
Bert 7 Topics: https://drive.google.com/file/d/1y6PPydfwmb47G_1x_2WNxvGcblvE5lkh/view?usp=sharing

