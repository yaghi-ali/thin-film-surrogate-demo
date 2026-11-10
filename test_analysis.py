import unittest
import numpy as np
from analysis import reflectance,train,predict
class Tests(unittest.TestCase):
    def test_zero_thickness_equals_interface(self):
        np.testing.assert_allclose(reflectance(np.array([450,600,750]),0),((1-1.52)/(1+1.52))**2,atol=1e-14)
    def test_index_matched_layer(self):
        np.testing.assert_allclose(reflectance(550,np.arange(100),n1=1.52),((1-1.52)/(1+1.52))**2,atol=1e-14)
    def test_held_out_generalization(self):
        r=np.random.default_rng(12);a=r.uniform([450,70],[750,150],(1000,2));b=r.uniform([450,70],[750,150],(300,2))
        ya=reflectance(a[:,0],a[:,1]);yb=reflectance(b[:,0],b[:,1]);state=train(a,ya);yp=predict(b,state)
        self.assertLess(np.sqrt(np.mean((yp-yb)**2)),1e-4)
        self.assertLess(np.mean((yp-yb)**2),np.mean((ya.mean()-yb)**2)/100)
    def test_invalid_wavelength(self):
        with self.assertRaises(ValueError):reflectance(0,100)
