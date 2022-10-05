## Fertilizers Recommendation System For Disease Prediction <img src="https://media1.giphy.com/media/f9Auu2zwYo1XGTsSPg/giphy.gif?cid=ecf05e47tnkhhm9of0gy1c66gntftbo34jm2iv8h5o4tq63e&rid=giphy.gif&ct=s" width="30px">
## Team:
    1. Sree Ram U [TL] - 312819205040
    
    2. Mohammed Farook C -312819205021
    
    3. Balaji M - 312819205006
    
    4. Pradeep V - 312819205029
## Introduction :
  - Agriculture is the most important sector in today’s life. Most plants are affected by a wide variety of bacterial and fungal diseases. Diseases on plants placed a major constraint on the production and a major threat to food security. Hence, early and accurate identification of plant diseases is essential to ensure high quantity and best quality. In recent years, the number of diseases on plants and the degree of harm caused has increased due to the variation in pathogen varieties, changes in cultivation methods, and inadequate plant protection techniques. 
  - An automated system is introduced to identify different diseases on plants by checking the symptoms shown on the leaves of the plant. Deep learning techniques are used to identify the diseases and suggest the precautions that can be taken for those diseases. 
## Technical Architecture : 
![image](https://user-images.githubusercontent.com/78264969/192721993-1527fbc3-0bcb-4523-8537-f023b98d789d.png)
## Project Objectives: 
        1. Preprocess the images.

        2. Applying the CNN algorithm to the dataset.

        3. How deep neural networks detect the disease.

        4. You will be able to know how to find the accuracy of the model.

        5. You will be able to build web applications using the Flask framework.   
## Project workflow :

>       * Project Progress 
>       
>           * [Problem Statement] - 
>           
>           * [Empathy Map] - https://github.com/IBM-EPBL/IBM-Project-26708-1660034316/tree/main/Empathy%20Map
>           
>           * [Literature Review] - 
## Prerequisites :
## Anaconda Navigator :
- Anaconda Navigator is a free and open-source distribution of the Python and R programming languages for data science and machine learning-related applications. It can be installed on Windows, Linux, and macOS. Conda is an open-source, cross-platform,  package management system. Anaconda comes with so very nice tools like JupyterLab, Jupyter Notebook,QtConsole, Spyder, Glueviz, Orange, Rstudio, Visual Studio Code. For this project, we will be using Jupiter notebook and spyder.To install Anaconda navigator and to know how to use Jupyter Notebook a Spyder using Anaconda watch the video given here.
 [https://youtu.be/5mDYijMfSzs]
 ### To build Deep learning models you must require the following packages :
 - Tensor flow: TensorFlow is an end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers can easily build and deploy ML powered applications.
 - Keras : Keras leverages various optimization techniques to make high level neural network API easier and more performant. It supports the following features.
 
       1. Consistent, simple and extensible API.
       
       2. Minimal structure - easy to achieve the result without any frills.
       
       3. It supports multiple platforms and backends.
       
       4. It is user-friendly framework that runs on both CPU and GPU.
       
       5. Highly scalability of computation.
 - Flask: Web framework used for building  Web applications.Watch the given below video to Install the necessary Packages.
 [https://youtu.be/akj3_wTploU]
 ## Project Structure :
 ![image](https://user-images.githubusercontent.com/78264969/193993919-df7c31a5-4d9e-4f67-b605-441597b45d92.png)
 - The dataset folder contains two folders for the fruit and vegetable dataset which again contains a test and train folder, each of them have images of different diseases.
- The Flask folder has all the files necessary to build the flask application. 

     1. Templates folder has the HTML pages.
  
     2. Uploads folder has the uploads made by the user.
  
     3. app.py is the python script for server-side computing.
  
     4. .h5 files are the model files that are to be saved after model building.
  
     5. Precautions excel files contain the precautions for all kinds of diseases.
  
 - Fruit-Training.ipynb, Vegetable-Training, and Plant-Disease-Testing.ipynb are the training and testing notebooks.
 - IBM folder contains IBM deployment files.
 
 Made with ❤ in India | Dept Of Information Technology | Agni College Of Technology
