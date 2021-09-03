# Comparision of Machine Learning Frameworks

## *Scikit Learn*
---

### Initial Release

**June, 2007**

### Written In Language

**Python, Cython, C and C++**

### Description

scikit-learn is a Python module for machine learning built on top of SciPy and distributed under the 3-Clause BSD license.

### Pros

- Scikit Learn provides a bunch of genuinely useful utilities for splitting data, computing common statistics, and doing even not-so-common matrix operations.
- Scikit Learn has good documentation, and a clean, mostly consistent API.
- Scikit-learn already implements a lot of non-neural net based algorithms that are commonly used in data science. It also offers a lot tools for data manipulations and utilities such as metrics functions, artificial dataset generations, and so on.
- Integrates well with Numpy and Pandas

### Cons
    
- Scikit Learn doesn’t use hardware acceleration making it slow at times; especially for training models.
- No Support for Deep Learning algorithms

---

## *Tensorflow*
---

### Initial Release

**November 9, 2015**

### Written In Language

**Python, C++, CUDA**

### Description

Created by Google and written in C++ and Python, TensorFlow is perceived to be one of the best open source libraries for numerical computation.TensorFlow is good for advanced projects, such as creating multi-layer neural networks.

### Pros

- It has a lot of documentation and guidelines
- It offers monitoring for training processes of the models and visualization  (Tensor board)
- It’s backed by a large community of devs and tech companies
- It provides model serving
- It supports distributed training

Chages in TensorFlow 2.0

The latest major version of the framework is TensorFlow 2.0. It brings us a bunch of exciting features, such as:

- Support for the Keras framework
- It is possible to use Keras inside Tensorflow. It ensures that new Machine Learning models can be built with ease.
- Supports debugging your graphs and networks - TensorFlow 2.0 runs with eager execution by default for ease of use and smooth debugging.
- Robust model deployment in production on any platform.
- Powerful experimentation for research.
- Simplifying the API by cleaning up deprecated APIs and reducing duplication.

### Cons
    
- It struggles with poor results for speed in [benchmark tests](https://arxiv.org/pdf/1608.07249v7.pdf) compared to other frameworks.

---

## *PyTorch*
---

### Initial Release

**October 2016**

### Written In Language

**Python, C++, LUA, CUDA**

### Description

PyTorch is the Python successor of Torch library written in Lua and a big competitor for TensorFlow. It was developed by Facebook. PyTorch is mainly used to train deep learning models quickly and effectively, so it’s the framework of choice for a large number of researchers.

### Pros

- The modeling process is simple and transparent thanks to the framework’s architectural style.
- The default define-by-run mode is more like traditional programming, and you can use common debugging tools as pdb, ipdb or PyCharm debugger.
- It has [declarative data parallelism](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b).
- It features a lot of retrained models and modular parts that are ready and easy to combine.
- It supports distributed training.
- Dynamic approach via GPU (Each level of computation can be viewed).
- Transparent to the user(Dynamic Graph outputs viewed faster).
- Easy to debug (Uses PyCharm for define-by-run mode during runtime).
- Data parallelism (Allows “torch.nn.DataParallel” to wrap any module).

### Cons

- It lacks model serving in production (Although it will change in the future)
- It lacks interfaces for monitoring and visualization such as Tensor board (As a workaround, you can connect externally to Tensor board)

---

## *Fast AI*
---

### Initial Release

**Oct 2, 2018**

### Written In Language

**Python**

### Description

fast.ai can be described as a research lab bundled with courses, an easy-to-use Python library with a huge community. Their library wraps popular deep learning and machine learning libraries for common workflows and provides a user-friendly interface. Most importantly, it follows the "top down" approach.

### Pros
 
- Much less code for you to write for most common tasks
- More best practices baked in, so normally faster to train and higher accuracy
- Easier to understand
- Handles tabular data much better
- Fits in with wider python ecosystem better (e.g pandas)
- The dynamic nature of pyTorch is much better for experimentation and iteration, and therefore many recent research papers are on pytorch first

### Cons
    
- Not much documentation
- Relies on pytorch, which doesn’t have such mature production (mobile or high scalability server) capabilities compared to tensorflow
- Pytorch doesn’t run on as many devices yet (e.g Google’s TPU)
- Not supported by as big an organization as tf
- Some parts still missing or incomplete (e.g object localization APIs)

---

## *DeepLearning4J*
---

### Initial Release

**August, 2016**

### Written In Language

**Java, Scala, CUDA, C, Clojure**

### Description

It’s a commercial-grade, open-source framework written mainly for Java and Scala, offering massive support for different types of neural networks (like CNN, RNN, RNTN, or LTSM). 

### Pros

- It’s robust, flexible and effective.
- It can process huge amounts of data without sacrificing speed.
- It works with Apache Hadoop and Spark, on top of distributed CPUs or GPUs.
- The documentation is really good.
- It has a community version and an enterprise version.

### Cons
    
- Java not a popular choice for AI/ML
- Training memory Limited by JVM heap size

---


## *Microsoft Cognitive Toolkit (Prev. CNTK)*
---

### Initial Release

**25 January, 2016**

### Written In Language

**C++**

### Description

This is now called The Microsoft Cognitive Toolkit – an open-source DL framework created to deal with big datasets and to support Python, C++, C#, and Java. CNTK facilitates really efficient training for voice, handwriting, and image recognition, and supports both CNNs and RNNs.

### Pros
- It delivers good performance and scalability;
- It features a lot of highly optimized components;
- It offers support for Apache Spark;
- It’s very efficient in terms of resource usage;
- It supports simple integration with Azure Cloud;

### Cons

- Limited community support.

---
