import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "level-1-learn"))
from signal_ml import haar_dwt, haar_idwt, extract_features, make_dataset, features_for_dataset, SmallNN, train

class DWTTests(unittest.TestCase):
    def test_dwt_known(self):
        a,d = haar_dwt(np.array([4.,6.,10.,12.]))
        self.assertTrue(np.allclose(a, [7.0710678118654755, 15.556349186104045]))
        self.assertTrue(np.allclose(d, [-1.4142135623730951, -1.4142135623730951]))
    def test_dwt_constant(self):
        a,d = haar_dwt(np.array([3.,3.,3.,3.]))
        self.assertTrue(np.allclose(d, [0.,0.])); self.assertTrue(np.allclose(a, [4.242640687119285,4.242640687119285]))
    def test_dwt_energy_preserved(self):
        x=np.array([1.,2.,3.,4.,5.,6.,7.,8.]); a,d=haar_dwt(x)
        self.assertLess(abs(np.sum(x*x)-(np.sum(a*a)+np.sum(d*d))),1e-9)
    def test_idwt_roundtrip(self):
        x=np.array([1.,-2.,0.5,3.,4.,0.,-1.,2.]); a,d=haar_dwt(x); self.assertTrue(np.allclose(haar_idwt(a,d),x))
    def test_dwt_odd_fails(self):
        with self.assertRaises(ValueError): haar_dwt(np.array([1.,2.,3.]))

class FeatureTests(unittest.TestCase):
    def test_features_shape(self):
        f=extract_features(np.sin(np.linspace(0,6.28,64))); self.assertEqual(f.shape,(12,)); self.assertTrue(np.all(np.isfinite(f)))
    def test_features_defect_higher_detail_energy(self):
        t=np.linspace(0,2*np.pi,64,endpoint=False); clean=np.sin(t)
        defect=clean.copy(); defect[20:25]+=1.2
        self.assertGreater(extract_features(defect)[5], extract_features(clean)[5])

class DatasetTests(unittest.TestCase):
    def test_dataset_deterministic(self):
        X1,y1=make_dataset(n_per_class=10,seed=0); X2,y2=make_dataset(n_per_class=10,seed=0)
        self.assertTrue(np.array_equal(X1,X2) and np.array_equal(y1,y2))
    def test_dataset_balanced_and_shape(self):
        X,y=make_dataset(n_per_class=20,seed=1); self.assertEqual(X.shape,(40,64)); self.assertEqual(int(np.sum(y==0)),20); self.assertEqual(int(np.sum(y==1)),20)
    def test_dataset_different_seed_differs(self):
        X1,_=make_dataset(n_per_class=5,seed=0); X2,_=make_dataset(n_per_class=5,seed=99); self.assertFalse(np.array_equal(X1,X2))

class NNTests(unittest.TestCase):
    def test_gradient_finite_difference(self):
        rng=np.random.default_rng(0); X=rng.normal(size=(8,12)); y=np.array([0,1,0,1,0,1,1,0])
        m=SmallNN(12,hidden=8,seed=2); loss,grads=m.loss_and_grads(X,y)
        eps=1e-6; orig=m.W2[0,0]; m.W2[0,0]=orig+eps; lp,_=m.loss_and_grads(X,y); m.W2[0,0]=orig-eps; lm,_=m.loss_and_grads(X,y); m.W2[0,0]=orig
        numeric=(lp-lm)/(2*eps); self.assertLess(abs(numeric-grads["W2"][0,0]),1e-5)
    def test_training_decreases_loss(self):
        Xsig,y=make_dataset(n_per_class=20,seed=0); X=features_for_dataset(Xsig)
        mu=X.mean(0); sd=X.std(0); Xn=(X-mu)/sd
        m=SmallNN(12,hidden=16,seed=1); l0,_=m.loss_and_grads(Xn,y)
        for _ in range(50):
            loss,g=m.loss_and_grads(Xn,y); m.step(g,lr=0.05)
        l1,_=m.loss_and_grads(Xn,y); self.assertLess(l1,l0)
    def test_train_accuracy_reasonable(self):
        Xtr_sig,ytr=make_dataset(n_per_class=50,seed=0); Xte_sig,yte=make_dataset(n_per_class=25,seed=7)
        Xtr=features_for_dataset(Xtr_sig); Xte=features_for_dataset(Xte_sig)
        _,met=train(Xtr,ytr,Xte,yte,epochs=100,lr=0.05,seed=1); self.assertGreaterEqual(met["accuracy"],0.8)
    def test_predict_shape(self):
        m=SmallNN(12,seed=1); p=m.predict(np.zeros((3,12))); self.assertEqual(p.shape,(3,))

if __name__=="__main__":
    unittest.main()
