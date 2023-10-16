from algorithm.utils.RobustSTL.RobustSTL import RobustSTL


def robuststl(y, T, reg1=1., reg2=0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.):
    _, trend, seasonal, resid = RobustSTL(y, T, reg1=reg1, reg2=reg2, K=K, H=H, dn1=dn1, dn2=dn2, ds1=ds1, ds2=ds2)
    return trend, seasonal, resid
