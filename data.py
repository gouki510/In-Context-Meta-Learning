from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
import numpy as np

class SamplingDataset(object):
  def __init__(self,conf):
    self.num_classes = conf.num_classes
    self.dim = conf.dim
    self.num_labels = conf.num_labels
    self.eps = conf.eps
    self.alpha = conf.alpha
    self.ways = conf.ways
    self.p_bursty = conf.p_bursty
    self.data_type = conf.data_type # "bursty", no_support, holdout, flip
    self.mu, self.labels = self._get_data()

  def _get_data(self):
    mu = torch.normal(mean=0, std=1/self.dim, size=(self.num_classes,self.dim))
    labels = torch.randint(self.num_labels, size=(self.num_classes,1))
    return mu, labels

class SamplingLoader(DataLoader):

  def __init__(self,conf):
    self.dataset = SamplingDataset(conf)
    self.mu, self.labels = self.dataset._get_data()
    self.data_type = conf.data_type
    self.num_seq = conf.num_seq
    self.alpha = conf.alpha
    self.num_classes = conf.num_classes
    self.num_labels = conf.num_labels
    self.ways = conf.ways
    self.p_bursty = conf.p_bursty
    self.eps = conf.eps
    self.dim = conf.dim
    self.num_holdout_classes = conf.num_holdout_classes
    self.holdout_classes = np.arange(self.num_classes-self.num_holdout_classes, self.num_classes)
    assert self.num_seq % self.ways ==0
    prob = np.array([1/(k+1)**self.alpha for k in range(self.num_classes-self.num_holdout_classes)])
    self.prob = prob/prob.sum()

  def get_seq(self):
    while True:
      if self.data_type=="bursty":
        if self.p_bursty > np.random.rand():
          # choise few shot example
          num_few_shot_class = self.num_seq//self.ways
          few_shot_class = np.random.choice(self.num_classes-self.num_holdout_classes, num_few_shot_class, replace=False)
          mus = self.mu[few_shot_class]
          mus = np.repeat(mus, self.ways, axis=0) # expand ways
          labels = self.labels[few_shot_class]
          labels = np.repeat(labels, self.ways, axis=0) # expand ways
          classes = np.repeat(few_shot_class, self.ways)
          # add noise
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]
          # select query labels
          query_class = np.random.choice(few_shot_class, 1)
          query_label = self.labels[query_class]
          query_mu = self.mu[query_class]
          query_x = self.add_noise(query_mu)
          # concat
          x = torch.cat([x, query_x])
          labels = torch.cat([labels.flatten(), torch.tensor(query_label).flatten()])
          yield {
              "examples":x.to(torch.float32),
              "labels":labels,
              "classes" : torch.cat([torch.tensor(classes).flatten(), torch.tensor(query_class).flatten()])
          }
        else:
          # rank frequency
          classes = np.random.choice(self.num_classes-self.num_holdout_classes, self.num_seq+1, p=self.prob)
          mus = self.mu[classes]
          labels = self.labels[classes]
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq+1)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]

          yield {
              "examples":x.to(torch.float32),
              "labels":labels.flatten(),
              "classes" : torch.tensor(classes)
          }

      elif self.data_type == "no_support":
          # rank frequency
          classes = np.random.choice(self.num_classes-self.num_holdout_classes, self.num_seq+1, p=self.prob, replace=False)
          mus = self.mu[classes]
          labels = self.labels[classes]
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq+1)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]

          yield {
              "examples":x.to(torch.float32),
              "labels":labels.flatten(),
              "classes" : torch.tensor(classes)
          }
          
      elif self.data_type == "holdout":
          # rank frequency
          classes = np.random.choice(self.holdout_classes, self.num_seq)
          mus = self.mu[classes]
          labels = self.labels[classes]
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]
          # query
          query_class = np.random.choice(classes, 1)
          query_label = self.labels[query_class]
          query_mu = self.mu[query_class]
          query_x = self.add_noise(query_mu)
          # concat
          x = torch.cat([x, query_x])
          labels = torch.cat([labels.flatten(), torch.tensor(query_label).flatten()])

          yield {
              "examples":x.to(torch.float32),
              "labels":labels,
              "classes" : torch.cat([torch.tensor(classes).flatten(), torch.tensor(query_class)])
          }

      elif self.data_type == "flip":
                # rank frequency
          classes = np.random.choice(self.num_classes, self.num_seq)
          mus = self.mu[classes]
          # label flip
          labels = (self.labels[classes] + 1) % self.num_labels
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]
          # query
          query_class = np.random.choice(classes, 1)
          query_label = (self.labels[query_class]+1) % self.num_labels
          query_mu = self.mu[query_class]
          query_x = self.add_noise(query_mu)
          # concat
          x = torch.cat([x, query_x])
          labels = torch.cat([labels.flatten(), torch.tensor(query_label).flatten()])
          yield {
              "examples":x.to(torch.float32),
              "labels":labels.flatten(),
              "classes" : torch.cat([torch.tensor(classes).flatten(), torch.tensor(query_class)])
          }

  def add_noise(self, x):
    x = (x+self.eps*np.random.normal(0, 1/self.dim, size=x.shape))/(np.sqrt(1+self.eps**2))
    return x

class IterDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator


    def __iter__(self):
        return self.generator()

