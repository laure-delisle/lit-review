# Learning with priors

*This is work in progress -- both braisntorming and reference gathering*

What are priors? Types of priors? (data structure, label structure, invariances, )


## List of references
Surveys
- [ ] Informed Machine Learning – A Taxonomy and Survey of Integrating Prior Knowledge into Learning Systems (von Rueden et al, IEEE KDEng 2021) [[pdf]](https://arxiv.org/pdf/1903.12394.pdf)
- [ ] Machine Learning with Prior Knowledge (Murphy, 2018) [[blogpost]](https://semiwiki.com/artificial-intelligence/7628-machine-learning-with-prior-knowledge/)
- [ ] Deep Learning, Structure and Innate Priors (See + Lecun/Manning, 2018) [[blogpost]](https://www.abigailsee.com/2018/02/21/deep-learning-structure-and-innate-priors.html)
- [ ] Structural Priors in Deep Neural Networks (Ioannou, 2017) [[thesis]](https://yani.ai/thesis_online.pdf)
- [ ] Integrating domain knowledge into deep learning: Increasing model performance through human expertise (Ståhl, 2021) [[thesis]](http://www.diva-portal.org/smash/get/diva2:1544786/FULLTEXT01.pdf)

Known Operator Learning
- [ ] Known Operator Learning - Towards Integration of Prior Knowledge into Machine Learning (Meier, Maths of DL 2019) [[youtube]](https://www.youtube.com/watch?v=ecQfRdtF0c4) [[blogpost]](https://towardsdatascience.com/known-operator-learning-part-1-32fc2ea49a9)

Human priors
- [ ] Expert-augmented machine learning (Gennatas et al, PNAS 2020) [[pdf]](https://www.pnas.org/content/pnas/117/9/4571.full.pdf)
- [ ] Integrating Machine Learning with Human Knowledge (Deng, iScience journal 2020) [[pdf]](https://reader.elsevier.com/reader/sd/pii/S2589004220308488?token=F3C27D14049AFCED7A9630C5533AA13C4BF092E8989064ABEE0856CCEBAE24F5AC8EFAC58067D040948F81975F707669&originRegion=us-east-1&originCreation=20220127220110)

Data priors
- [ ] Neural Blind Deconvolution Using Deep Priors (Ren et al, CVPR 2020) [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ren_Neural_Blind_Deconvolution_Using_Deep_Priors_CVPR_2020_paper.pdf)
- [ ] Structured Weight Priors for Convolutional Neural Networks (Pearce et al, ICML workshop Uncertainty/robustness 2020) [[pdf]](https://arxiv.org/pdf/2007.14235.pdf)
- [ ] Data as Prior/Innate knowledge for Deep Learning models (Amatriain, 2019) [[blogpost]](https://xamat.medium.com/data-as-prior-innate-knowledge-for-deep-learning-models-23898363a71a)
- [ ] Encode prior knowledge in deep neural networks (Aditya et al., 2018) [[quora]](https://www.quora.com/Is-there-a-way-to-encode-prior-knowledge-in-deep-neural-networks)


Mixed human and data priors
- [ ] Incorporating Prior Domain Knowledge into Deep Neural Networks (Muralidhar, IEEE Big Data 2018) [[pdf]](https://people.cs.vt.edu/ramakris/papers/PID5657885.pdf)

Bayesian priors:
- [ ] Learning From the Experience of Others: Approximate Empirical Bayes in Neural Networks (Zhao et al, ICLR 2019, REJECTED) [[openreview]](https://openreview.net/forum?id=r1E0OsA9tX) [[pdf]](https://openreview.net/pdf?id=r1E0OsA9tX)
	- problems with the approach: "Deterministic Latent Variable Models and their Pitfalls" (Welling et al. 2008)
	- [refs] use of non-conjugate likelihoods in empirical Bayes:  [Emp Bayes method to optim ML algos](https://papers.nips.cc/paper/6864-an-empirical-bayes-approach-to-optimizing-machine-learning-algorithms.pdf), [Recasting Gradient-Based Meta-Learning as Hierarchical Bayes](https://arxiv.org/abs/1801.08930), [Conditional Neural Processes](https://arxiv.org/abs/1807.01613)