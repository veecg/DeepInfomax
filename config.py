import os
from datetime import datetime


SEED = 123

ALPHA = 0 #weight for global objective
BETA = 1.0  #weight for local objective
GAMMA = 0.1 #weight for prior matching objective

LEARNING_RATE = 1e-5
EPOCH_DIM = 500 
EPOCH_CLS = 200
BATCH_SIZE = 64


HOME_PATH = os.path.abspath(os.path.dirname(__file__))


def folder_creation(PATH, mode):
    """
    Setup proper folder for each training process
    PATH:   path under which to create the folder
        'train':    pass the home path
        'log':      pass the training folder path
        'checkpoint':    pass the training folder path

    Mode option:
        'train':            Create a training folder with filename as current timestamp. 
            'log':          Create a log folder as the default logdir for tensorboard and logging output
            'checkpoint':    Create a folder to store the trained models parameters
    """
    if mode=='train':
        mydir = os.path.join(
            PATH,
            'output',
            'A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_'+str(datetime.now().strftime('%m%d_%H%M')))
    elif mode=='log':
        mydir = os.path.join(PATH,'log')
    elif mode=='checkpoint':
        mydir = os.path.join(PATH,'checkpoint')
    elif mode=='clustering':
        mydir = os.path.join(PATH,'task_result','clustering',str(datetime.now().strftime('%m%d_%H%M')))
    elif mode=='task1':
        mydir = os.path.join(PATH,'task_result','classification','fss'+str(datetime.now().strftime('%m%d_%H%M')))
    elif mode=='task2':
        mydir = os.path.join(PATH,'task_result','classification','conv'+str(datetime.now().strftime('%m%d_%H%M')))
    elif mode=='task3':
        mydir = os.path.join(PATH,'task_result','classification','fc'+str(datetime.now().strftime('%m%d_%H%M')))
    elif mode=='task4':
        mydir = os.path.join(PATH,'task_result','classification','Y'+str(datetime.now().strftime('%m%d_%H%M')))
    
    else:
        raise ValueError("Wrong mode is given in folder_creation.") 
    
    try:
        os.makedirs(mydir)
        print("Folder created at:", mydir)
    except FileExistsError:
        print("Folder already created at:", mydir)
    finally:
        return mydir
        
    
