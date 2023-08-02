import torch 
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture as GM
from torch.utils.data import DataLoader
from comet_ml.integration.sklearn import log_model as log_model_sklearn
from comet_ml.integration.pytorch import log_model as log_model_pytorch

class AD_GMM():
    """ a class that define the gmm training and prediction for anomaly detection
    uses model for feature extraction"""
    def __init__(self, num_classes:int, model):
        self.gmm = GM(n_components=num_classes, covariance_type='full')
        self.model = model
    
    def fit(self, dataloader):
        features = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                data, target = batch
                _, feature = self.model(data)
                feature = feature.detach().numpy()
                features.append(feature)
        features = np.concatenate(features)
        self.gmm.fit(features)
    
    def predict(self, data):
        if data.ndim == 1:
            data = data.reshape(1,-1)
        _, feature = self.model(data)
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
    
