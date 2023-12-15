import torch 
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture as GM
from comet_ml.integration.sklearn import log_model as log_model_sklearn
from comet_ml.integration.pytorch import log_model as log_model_pytorch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from tqdm import tqdm

# define an abstract class for anomaly detection system 
# so that we can use the same interface for all the systems
# predict and fit methods are mandatory
class AD_system(ABC):
    @abstractmethod
    def fit(self,data):
        pass
    
    @abstractmethod
    def predict(self,data):
        pass
    
    @abstractmethod
    def save(self,filename):
        pass
    
    @classmethod
    @abstractmethod
    def load(cls,filename):
        pass
    
    @classmethod
    @abstractmethod
    def load_from_log(cls,logger):
        pass
    
    @abstractmethod
    def log_model(self,logger):
        pass



class AD_GMM(AD_system):
    """ a class that define the gmm training and prediction for anomaly detection
    uses model for feature extraction"""
    def __init__(self, num_classes:int, model):
        self.gmm = GM(n_components=num_classes, covariance_type='full')
        self.model = model
    
    def load_all(self, dataloader):
        features = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Loading data"):
                data = batch
                feature,_,_ = self.model(data)
                feature = feature.detach().numpy()
                features.append(feature)
        features = np.concatenate(features)
        return features
    
    
    def fit(self, dataloader):
        features = self.load_all(dataloader)

        self.gmm.fit(features)
    
    def predict(self, data):
        if data.ndim == 1:
            data = data.reshape(1,-1)
        self.model.eval()
        feature,_,_ = self.model(data)
        feature = feature.detach().numpy()
        log_likelihood = self.gmm.score_samples(feature)
        return log_likelihood

    def save(self, filename):
        # Save pytorch model
        torch.save(self.model.state_dict(), filename + ".pth")
        # Save GMM model
        joblib.dump(self.gmm, filename + "_gmm.joblib")

    def log_model(self,logger):
        log_model_pytorch(logger,model=self.model, model_name="model_nn")
        log_model_sklearn(logger,model=self.gmm, model_name="gmm")

    def find_best_ncomp(self,dataloader, n_components):
        feature = self.load_all(dataloader)
        res =dict()
        for n_comp in n_components:
            gmm = GM(n_components=n_comp, covariance_type='full')
            gmm.fit(feature)
            bic = gmm.bic(feature)
            res[n_comp] = bic
            print(f"n_comp: {n_comp}, bic: {bic}")
        # make a plot of the bic
        plt.figure()
        plt.plot(res.keys(),res.values())
        plt.show()
        return res
            
    @classmethod
    def load(cls, filename, num_classes, model_class, *model_args):
        # Load pytorch model
        model = model_class(*model_args)
        model.load_state_dict(torch.load(filename + ".pth"))
        model.eval()
        # Load GMM model
        gmm = joblib.load(filename + "_gmm.joblib")
        
        # Initialize AD_GMM with loaded models
        ad_gmm = cls(num_classes, model)
        ad_gmm.gmm = gmm
        return ad_gmm
    

    @classmethod
    def load_from_log(cls, logger, num_classes):
        model = logger.experiment.get_model(prefix="model")
        gmm = logger.experiment.get_model(prefix="gmm")
        model.eval()
        ad_gmm = cls(num_classes, model)
        ad_gmm.gmm = gmm
        return ad_gmm
    
