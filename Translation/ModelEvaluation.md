Source: https://en.wikipedia.org/wiki/Precision_and_recall

    以下是关于模型评估的一些指标，首先假设测试集里面包含10个样本，其中7个正例，3个反例，模型预测出来的结果中包含6个正例和4个反例，

其中6个正例中有5个是真正的正例还有一个是预测错误，实际上是反例，这个5就是真正例TP，那么可以计算出FP=6-TP=1,FN=7-TP=2,TN=4-FN=2

    precision(查准率，准确率): TP/(TP+FP) = 5/6   -> 简写为P
    
    recall(查全率，召回率,TPR): TP/(TP+FN) = 5/7  -> ROC曲线的纵轴，简写为R
 
    Accuracy(精度): (TP + TN) / (sample count) = 7/10 
    
    FPR: FP/(FP+TN) = 1/3    -> ROC曲线的横轴
    
    F = 2PR/(P+R)  -> F为P和R的调和平均，即 1/F = 1/2(1/P + 1/R)
    
    F-beta = (1+beta**2)PR/(beta**2*P + R) -> F的一般形式，beta=1时为F
    
    当有多个2分类矩阵时，比如多个测试集/训练集，或者是个多分类任务，然后我们两两计算混淆矩阵，得到多组查准率和查全率
    
    (P1,R1) (P2,R2)...(Pn,Rn)
    
    这个时候我们可以通过两种逻辑来计算它们的平均值：
    
    1. macro方式
    
    macro-P = 1/n(P1+P2+...+Pn)
    
    macro-R = 1/n(R1+R2+...+Rn)
    
    macro-F = 2*macro-P*macro-R/(macro-P + macro-R)
    
    2. micro方式, 先将各混淆矩阵的元素进行平均，得到mean(TP),mean(TN),mean(FP),mean(FN)
    
    micro-P = mean(TP)/[mean(TP)+mean(FP)]
    
    micro-R = mean(TP)/[mean(TP)+mean(FN)]
    
    micro-F = 2*micro-P*micro-R/(micro-P + micro-R)
    
    主要注意的有以下几点：
    
    查准率和查全率是两者不可兼得的，提高其中之一则必定降低另外一个，查准率意为预测的正例中有多少个是真正例，而查全率意味着所有的正例中有多少个被模型成功预测出来了。就好比一个外科医生给患者做肿瘤切除手术，一个激进的医生可能更注重切除所有的肿瘤细胞，甚至不惜切除健康的细胞，这就是提高了查全率，而降低了精确率。另一个保守的医生可能只关注切除的所有组织里只包含肿瘤细胞，尽量不要切除健康的组织，这就是关注了查准率，但很可能会遗漏掉肿瘤细胞，降低查全率
