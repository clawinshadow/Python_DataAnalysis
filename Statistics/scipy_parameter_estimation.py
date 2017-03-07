import scipy.stats as st
import math

def me(sv, sm, n, alpha, bilateral=True):
    """
    Calculate the interval estimation of population mean.
    Assume population is a normalized distribution, std variance of popolation
    is known(正态总体，总体标准差已知，用样本均值来估计总体均值):
          sm - pm
    z = ------------ ~ N(0, 1)
        sv / sqrt(n)

    Parameters
    ----------
    sm : sample mean, 样本均值
    n  : sample count, 样本容量
    sv : population/sample standard variance, 总体或样本标准差
         当总体服从正态分布，但标准差未知，或者任意分布但是大样本前提下，
         均可以用样本标准差来代替总体标准差
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pm1, pm2) : interval estimation of population mean value,
                      总体均值的区间估计
    """

    rv = st.norm(0, 1)
    z = 0
    if bilateral:
        z = rv.ppf((1 + alpha) / 2)
    else:
        z = rv.ppf(alpha)
    return tuple([sm - z * sv / math.sqrt(n), sm + z * sv / math.sqrt(n)])


def me2(sv, sm, n, alpha, bilateral=True):
    """
    Calculate the interval estimation of population mean.
    Assume population is a normalized distribution, std variance of popolation
    is unknown, small sampling
    (正态总体，总体标准差未知，小样本情况下，用样本均值来估计总体均值):
          sm - pm
    z = ------------ ~ t(n-1)
        sv / sqrt(n)

    Parameters
    ----------
    sm : sample mean, 样本均值
    n  : sample count, 样本容量
    sv : sample standard variance, 样本标准差
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pm1, pm2) : interval estimation of population mean value,
                      总体均值的区间估计
    """

    rv = st.t(n-1)
    z = 0
    if bilateral:
        z = rv.ppf((1 + alpha) / 2)
    else:
        z = rv.ppf(alpha)
    return tuple([sm - z * sv / math.sqrt(n), sm + z * sv / math.sqrt(n)])


def pe(sp, n, alpha, bilateral=True):
    """
    Calculate the interval estimation of population proportion.
    Assume it's a large sampling
    (大样本前提下，用样本比例来估计总体比例):
            sp - pi
    z = ---------------- ~ N(0, 1)
         sqrt(p(1-p)/n)

    Parameters
    ----------
    sp : sample proportion, 样本比例
    n  : sample count, 样本容量
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pi1, pi2) : interval estimation of population proportion,
                      总体比例的区间估计
    """

    rv = st.norm(0, 1)
    z = 0
    if bilateral:
        z = rv.ppf((1 + alpha) / 2)
    else:
        z = rv.ppf(alpha)
        
    sv = math.sqrt(sp * (1 - sp) / n)    
    return tuple([sp - z * sv, sp + z * sv])


def ve(sv, n, alpha, bilateral=True):
    """
    Calculate the interval estimation of population variance.
    Assume it's a normalized population
    (总体方差的区间估计，只考虑正态总体的估计问题):
      (n - 1)* sv**2
    ------------------ ~ Chi-square(n - 1)
           pv**2

    Parameters
    ----------
    sv : sample proportion, 样本标准差
    n  : sample count, 样本容量
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pv1, pv2) : interval estimation of population variance,
                      总体方差的区间估计
    """

    rv = st.chi2(n - 1)
    z1 = rv.ppf((1 - alpha) / 2)
    z2 = rv.ppf((1 + alpha) / 2)
             
    return tuple([(n - 1) * sv**2/z2, (n - 1) * sv**2/z1])


def mediff(sm1, sm2, sv1, sv2, n1, n2, alpha, bilateral=True):
    """
    Calculate the interval estimation of the difference of 2 population mean values
    Assume 2 populations are normalized distributions,
    or they are both large sampling
    (如果两个总体均为正态分布，或不为正态分布但都是大样本):
          (sm1 - sm2) - (pm1 - pm2)
    z = ---------------------------- ~ N(0, 1)
         sqrt(sv1**2/n1 + sv2**2/n2)

    Parameters
    ----------
    sm1 : sample mean, 总体1的样本均值
    sm2 : sample mean, 总体2的样本均值
    n1  : sample count, 总体1的样本容量
    n2  : sample count, 总体2的样本容量
    sv1 : sample standard variance, 总体1的样本标准差或是总体标准差
    sv2 : sample standard variance, 总体2的样本标准差或是总体标准差
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pm1, pm2) : interval estimation of diff of population mean value,
                      总体均值之差的区间估计
    """

    rv = st.norm(0, 1)
    z = 0
    if bilateral:
        z = rv.ppf((1 + alpha) / 2)
    else:
        z = rv.ppf(alpha)

    smdiff = sm1 - sm2
    svsum = math.sqrt(sv1**2/n1 + sv2**2/n2)
    # print(svsum)
    return tuple([smdiff - z * svsum, smdiff + z * svsum])


def mediff2(sm1, sm2, sv1, sv2, n1, n2, alpha, bilateral=True):
    """
    Calculate the interval estimation of the difference of 2 population mean values
    Assume 2 populations are normalized distributions,
    or they are both large sampling
    (如果两个总体的方差未知但相等，且都为小样本情况下):
    
              (n1 - 1)*sv1**2 + (n2 - 1)*sv2**2
    Sp**2 = --------------------------------------
                         n1 + n2 - 2
    
          (sm1 - sm2) - (pm1 - pm2)
    t = ----------------------------- ~ t(n1 + n2 - 2)
            Sp * sqrt(1/n1 + 1/n2)

    Parameters
    ----------
    sm1 : sample mean, 总体1的样本均值
    sm2 : sample mean, 总体2的样本均值
    n1  : sample count, 总体1的样本容量
    n2  : sample count, 总体2的样本容量
    sv1 : sample standard variance, 总体1的样本标准差
    sv2 : sample standard variance, 总体2的样本标准差
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pm1, pm2) : interval estimation of diff of population mean value,
                      总体均值之差的区间估计(小样本且总体标准差未知但相等)
    """

    df = n1 + n2 - 2
    rv = st.t(df)
    sp = math.sqrt(((n1 - 1)*sv1**2 + (n2 - 1)*sv2**2)/df)
    z = 0
    if bilateral:
        z = rv.ppf((1 + alpha) / 2)
    else:
        z = rv.ppf(alpha)

    smdiff = sm1 - sm2
    svsum = sp * math.sqrt(1/n1 + 1/n2)
    # print(svsum)
    return tuple([smdiff - z * svsum, smdiff + z * svsum])


def mediff3(sm1, sm2, sv1, sv2, n1, n2, alpha, bilateral=True):
    """
    Calculate the interval estimation of the difference of 2 population mean values
    Assume 2 populations are normalized distributions,
    or they are both large sampling
    (如果两个总体的方差未知且不相等，且都为小样本情况下):

                   (sv1**2/n1 + sv2**2/n2)**2
    df = -------------------------------------------------
           (sv1**2/n1)**2/(n1-1) + (sv2**2/n2)**2/(n2-1)
    
            (sm1 - sm2) - (pm1 - pm2)
    t = --------------------------------- ~ t(df)
           sqrt(sv1**2/n1 + sv2**2/n2)

    Parameters
    ----------
    sm1 : sample mean, 总体1的样本均值
    sm2 : sample mean, 总体2的样本均值
    n1  : sample count, 总体1的样本容量
    n2  : sample count, 总体2的样本容量
    sv1 : sample standard variance, 总体1的样本标准差
    sv2 : sample standard variance, 总体2的样本标准差
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pm1, pm2) : interval estimation of diff of population mean value,
                      总体均值之差的区间估计(小样本且总体标准差未知且不相等)
    """

    df = math.floor(((sv1**2/n1 + sv2**2/n2)**2)/((sv1**2/n1)**2/(n1-1) + (sv2**2/n2)**2/(n2-1)))
    rv = st.t(df)
    z = 0
    if bilateral:
        z = rv.ppf((1 + alpha) / 2)
    else:
        z = rv.ppf(alpha)

    smdiff = sm1 - sm2
    svsum = math.sqrt(sv1**2/n1 + sv2**2/n2)
    # print(svsum)
    return tuple([smdiff - z * svsum, smdiff + z * svsum])


def mediff4(dm, sd, n, alpha, bilateral=True):
    """
    Calculate the interval estimation of the difference of 2 population mean values
    Assume matched sample(匹配样本的情况下，估计两个总体的均值之差):
          dm - mediff
    z = --------------- ~ N(0, 1)
          sd / sqrt(n)

    Parameters
    ----------
    dm : sample mean, 两个样本各均值之差的均值
    n  : sample count, 样本容量
    sd : population/sample standard variance, 总体或样本之差的标准差
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pm1, pm2) : interval estimation of population mean value,
                      总体均值的区间估计(匹配样本的情况下)
    """

    rv = st.norm(0, 1)
    z = 0
    if bilateral:
        z = rv.ppf((1 + alpha) / 2)
    else:
        z = rv.ppf(alpha)
    return tuple([dm - z * sd / math.sqrt(n), dm + z * sd / math.sqrt(n)])


def mediff5(dm, sd, n, alpha, bilateral=True):
    """
    Calculate the interval estimation of the difference of 2 population mean values
    Assume matched sample(匹配样本且为小样本的情况下，估计两个总体的均值之差):
          dm - mediff
    z = --------------- ~ t(n-1)
          sd / sqrt(n)

    Parameters
    ----------
    dm : sample mean, 两个样本各均值之差的均值
    n  : sample count, 样本容量
    sd : population/sample standard variance, 样本之差的标准差
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pm1, pm2) : interval estimation of population mean value,
                      总体均值的区间估计(匹配样本且为小样本的情况下)
    """

    rv = st.t(n-1)
    z = 0
    if bilateral:
        z = rv.ppf((1 + alpha) / 2)
    else:
        z = rv.ppf(alpha)
    return tuple([dm - z * sd / math.sqrt(n), dm + z * sd / math.sqrt(n)])


def pediff(sp1, sp2, n1, n2, alpha, bilateral=True):
    """
    Calculate the interval estimation of the difference of 2 population proportion
    
                  (sp1 - sp2) - (pp1 - pp2)
    z = ----------------------------------------------- ~ N(0, 1)
         sqrt(sp1 * (1-sp1) / n1 + sp2 * (1-sp2) / n2)

    Parameters
    ----------
    sp1 : sample mean, 总体1的样本比例
    sp2 : sample mean, 总体2的样本比例
    n1  : sample count, 总体1的样本容量
    n2  : sample count, 总体2的样本容量
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pm1, pm2) : interval estimation of diff of population proportion,
                      总体比例之差的区间估计
    """

    rv = st.norm(0, 1)
    z = 0
    if bilateral:
        z = rv.ppf((1 + alpha) / 2)
    else:
        z = rv.ppf(alpha)

    smdiff = sp1 - sp2
    svsum = math.sqrt(sp1*(1-sp1)/n1 + sp2*(1-sp2)/n2)
    # print(svsum)
    return tuple([smdiff - z * svsum, smdiff + z * svsum])


def sediff(sv1, sv2, n1, n2, alpha, bilateral=True):
    """
    Calculate the interval estimation of the difference of 2 population variance
    (两个总体方差比的区间估计)
    
    sv1**2    pv2**2 
    ------ * -------- ~ F(n1-1, n2-1)
    sv2**2    pv1**2

    Parameters
    ----------
    sv1 : sample mean, 总体1的样本标准差
    sv2 : sample mean, 总体2的样本标准差
    n1  : sample count, 总体1的样本容量
    n2  : sample count, 总体2的样本容量
    alpha: confidence level, 置信水平
    bilateral: 是否是双侧检验
    
    Returns
    -------
    tuple(pm1, pm2) : interval estimation of diff of population proportion,
                      两个总体方差比的区间估计
    """

    rv = st.f(n1-1, n2-1)
    f1 = rv.ppf((1 - alpha) / 2)
    f2 = rv.ppf((1 + alpha) / 2)

    svdiff = sv1**2/sv2**2
    # print(svsum)
    return tuple([svdiff/f2, svdiff/f1])


def nm(sv, alpha, expect):
    """
    估计总体均值时样本量的确定：n = sv**2 * Z(alpha)**2 / expect**2
    """

    rv = st.norm(0, 1)
    z = rv.ppf((1 + alpha) / 2)
    return math.ceil(sv**2 * z**2 / expect**2)


def np(pi, alpha, expect):
    """
    估计总体比例时样本量的确定：n = pi*(1-pi) * Z(alpha)**2 / expect**2
    """

    rv = st.norm(0, 1)
    z = rv.ppf((1 + alpha) / 2)
    return math.ceil(pi * (1 - pi) * z**2 / expect**2)


print(me.__doc__)
print(me(12, 81, 100, 0.95))
print(me2(4, 33, 5, 0.95))
print(pe(0.2, 500, 0.95))
print(ve(math.sqrt(93.21), 25, 0.95))
print(mediff(150, 140, 6, math.sqrt(24), 35, 35, 0.95))
print(mediff(1.0, 0.9, 1.1, 1.1, 91, 86, 0.9))
print(mediff2(32.5, 28.8, math.sqrt(15.996), math.sqrt(19.358), 12, 12, 0.95))
print(mediff3(32.5, 27.875, math.sqrt(15.996), math.sqrt(23.014), 12, 8, 0.95))
print(mediff5(11, 6.53, 10, 0.95))
print(pediff(0.45, 0.32, 500, 400, 0.95))
print(sediff(math.sqrt(260), math.sqrt(280), 25, 25, 0.9))
print(nm(2000, 0.95, 400))
print(np(0.9, 0.95, 0.05))
print(pediff(40/91, 21/86, 91, 86, 0.95))
print(nm(40, 0.95, 5))
print(nm(15, 0.95, 5))
print(me(4, 25.3, 60, 0.95))
print(me(6, 32, 50, 0.95))
print(me(2200, 12168, 480, 0.95))
print(me2(3.3, 17.25, 20, 0.95))
print(me(5, 22.4, 61, 0.95))
print(nm(80, 0.95, 15))
print(pe(0.26, 400, 0.95))
print(me(8.2, 52, 650, 0.95))
print(mediff2(15700, 14500, 700, 850, 8, 12, 0.95))
print(pediff(63/150, 60/200, 150, 200, 0.95))
