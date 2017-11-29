# Анализ гистологических снимков печени
Данный репозиторий подготовлен в рамках выполнения тестовой задачи от компании UNIM. Целью тестового задания является создание программной модели, которая на основе предоставленных гистологических снимков “научится” классифицировать гистологические снимки печени. Модель должна уметь по снимку печени делать вывод о том, к какому классу нужно отнести изображение - bad (“жир”, печень пораженную неалкогольным ожирением) или good (“норма”, печень без признаков поражения).
## Структура каталога проекта
| Относительный путь | Описание |
| :--- | :--- |
| _**./sctipts/fat_detector_cv.py**_ |  Скрипт с реализацией handmade-алгоритма классификации снимков.|
| _**./sctipts/augmentation.py**_ |  Скрипт, служащий для генерации тайлов и аугментация в рамках нейросетевого подхода к решению задачи.|
| _**./sctipts/loader.py**_ |  Python-генератор, используемый для обучения нейросети с использованием метода fit_generator из Keras. |
| _**./sctipts/fat_detector_neuro.py**_ |  Скрипт для создания модели и обучения нейросети. |
| _**./sctipts/install.sh**_ |  Скрипт для развертывания окружения и установки пакетов. |
| _**./sctipts/validation.ipynb**_ |  Jupyter-ноутбук, содержащий результаты тестирования алгоритмов на изображениях высокого разрешения. |
## Используемые наборы данных
Подготовленные в ходе проекта данные можно скачать в виде zip-архива (~ 11Gb) [по ссылке](https://yadi.sk/d/Kta4z2Mr3QAFwo). Скаченный архив следует развернуть в корневом каталоге проекта. 
### Структура каталога с данными
| Относительный путь | Описание |
| :--- | :--- |
| _**./data/test_bad/**_ |  Каталог с исходными изображениями пораженной печени, используемыми для тестирования и валидации. |
| _**./data/test_good/**_ |  Каталог с исходными изображениями здоровой печени, используемыми для тестирования и валидации. |
| _**./data/train_bad/**_ |  Каталог с исходными изображениями пораженной печени, используемыми для обучения. |
| _**./data/train_good/**_ |  Каталог с исходными изображениями здоровой печени, используемыми для обучения. |
| _**./data/train/bad/**_ | Сгенерированные тайлы пораженной печени, используемые для обучения нейросети. |
| _**./data/train/good/**_ | Сгенерированные тайлы здоровой печени, используемые для обучения нейросети. |
| _**./data/test/bad/**_ | Сгенерированные тайлы пораженной печени, используемые для тестирования на этапе обучения нейросети. |
| _**./data/test/good/**_ | Сгенерированные тайлы здоровой печени, используемые для тестирования на этапе обучения нейросети. |

## Веса модели
Для вычисления итоговых метрик качества работы алгоритма использовались веса с эпохи 23. Эти веса находятся в файле [weights.hdf5](https://yadi.sk/d/lZax9kT63Q5yqh). Данный файл необходимо разместить в каталоге проекта по следующему пути _**./sctipts/weights.hdf5**_