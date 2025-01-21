from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
import numpy as np
import math

class SamplingDataset(object):
  def __init__(self,conf):
    self.num_classes = conf.num_classes
    self.dim = conf.dim
    self.num_labels = conf.num_labels
    self.mu, self.labels = self._get_data()

  def _get_data(self):
    mu = torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(self.num_classes,self.dim))
    labels = torch.randint(self.num_labels, size=(self.num_classes,1))
    return mu, labels

class SamplingLoader(DataLoader):

  def __init__(self,conf, dataset):
    self.dataset = dataset
    self.mu, self.labels = self.dataset.mu, self.dataset.labels
    self.data_type = conf.data_type
    self.num_seq = conf.num_seq
    self.alpha = conf.alpha
    self.num_classes = conf.num_classes
    self.num_labels = conf.num_labels
    self.ways = conf.ways
    self.p_bursty = conf.p_bursty
    self.p_icl = conf.p_icl
    self.eps = conf.eps
    self.dim = conf.dim
    if self.ways != 0:
      assert self.num_seq % self.ways == 0
    if self.ways == 0:
      self.p_bursty = 0
    prob = np.array([1/((k+1)**self.alpha) for k in range(self.num_classes)])
    self.prob = prob/prob.sum()

  def get_seq(self):
    while True:
      if self.data_type=="bursty":
        if self.p_icl > np.random.rand():
            # choise few shot example
            num_few_shot_class = self.num_seq//self.ways
            mus, labels = self._get_novel_class_seq(num_few_shot_class)
            # mus = self.mu[few_shot_class]
            mus = np.repeat(mus, self.ways, axis=0) # expand ways
            # labels = self.labels[few_shot_class]
            labels = np.repeat(labels, self.ways, axis=0) # expand ways
            classes = np.arange(num_few_shot_class)
            classes = np.repeat(classes, self.ways)
            # add noise
            x = self.add_noise(mus)
            # permutation shuffle
            ordering = np.random.permutation(self.num_seq)
            mus = mus[ordering]
            x = x[ordering]
            labels = labels[ordering]
            classes = classes[ordering]
            # select query labels
            query_class_idx = np.random.choice(len(classes), 1)
            query_class = classes[query_class_idx]
            query_label = labels[query_class_idx]
            query_mu = mus[query_class_idx]
            query_x = self.add_noise(query_mu)
            # concat
            x = torch.cat([x, query_x])
            labels = torch.cat([labels.flatten(), query_label.flatten()])
            
            yield {
                "examples":x.to(torch.float32),
                "labels":labels,
                "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
            }
            
        else:
          if self.p_bursty > np.random.rand():
            # choise few shot example
            num_few_shot_class = self.num_seq//self.ways
            few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
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
            labels = torch.cat([labels.flatten(), query_label.flatten()])
            yield {
                "examples":x.to(torch.float32),
                "labels":labels,
                "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
            }
            
          else:
            # rank frequency
            classes = np.random.choice(self.num_classes, self.num_seq+1, p=self.prob)
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
                "classes" : torch.from_numpy(classes)
            }

      elif self.data_type == "no_support":
          # rank frequency
          classes = np.random.choice(self.num_classes, self.num_seq+1, p=self.prob)
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
              "classes" : torch.from_numpy(classes)
          }
          
      elif self.data_type == "holdout":
        # choise few shot example
        num_few_shot_class = self.num_seq//self.ways
        mus, labels = self._get_novel_class_seq(num_few_shot_class)
        # mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.ways, axis=0) # expand ways
        # labels = self.labels[few_shot_class]
        labels = np.repeat(labels, self.ways, axis=0) # expand ways
        classes = np.arange(num_few_shot_class)
        classes = np.repeat(classes, self.ways)
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        mus = mus[ordering]
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        # select query labels
        query_class_idx = np.random.choice(len(classes), 1)
        query_class = classes[query_class_idx]
        query_label = labels[query_class_idx]
        query_mu = mus[query_class_idx]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        
        yield {
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }

      elif self.data_type == "flip":
        # choise few shot example
        num_few_shot_class = self.num_seq//self.ways
        few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
        mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.ways, axis=0) # expand ways
        classes = np.repeat(few_shot_class, self.ways)
        # label flip
        labels = (self.labels[classes] + 1) % self.num_labels
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        # select query labels
        query_class = np.random.choice(few_shot_class, 1)
        query_label = (self.labels[query_class] + 1) % self.num_labels
        query_mu = self.mu[query_class]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        
        yield {
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }
    
  

  def add_noise(self, x):
    x = (x+self.eps*torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(x.shape)))/(np.sqrt(1+self.eps**2))
    # x = (x+self.eps*np.random.normal(mean=0, std=np.sqrt(1/self.dim), size=(x.shape[0],1)))/(np.sqrt(1+self.eps**2))
    return x
  
  def _get_novel_class_seq(self,num_class):
    mu = torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(num_class,self.dim))
    labels = torch.randint(self.num_labels, size=(num_class,1))
    return mu, labels

class IterDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()
      
class IterDatasetFortask(IterableDataset):
    def __init__(self, generator):
        self.generator = generator
    def __iter__(self):
        return self.generator()


    
class MultiTaskSamplingLoader(DataLoader):

  def __init__(self,conf, dataset):
    self.dataset = dataset
    self.mu, self.labels = self.dataset.mu, self.dataset.labels
    self.data_type = conf.data_type
    self.num_seq = conf.num_seq
    self.alpha = conf.alpha
    self.num_classes = conf.num_classes
    self.num_task = conf.num_tasks
    self.num_labels = conf.num_labels
    self.task_ways = conf.task_ways
    self.item_ways = conf.item_ways
    self.p_bursty = conf.p_bursty
    self.p_icl = conf.p_icl
    self.eps = conf.eps
    self.dim = conf.dim
    self.task_alpha = conf.task_alpha
    if self.item_ways != 0 or self.task_ways != 0:
      assert self.num_seq % self.item_ways == 0 and self.num_seq % self.task_ways == 0
    if self.item_ways == 0 or self.task_ways == 0:
      self.p_bursty = 0
    prob = np.array([1/((k+1)**self.alpha) for k in range(self.num_classes)])
    self.prob = prob/prob.sum()
    task_alpha = self.alpha
    task_prob = np.array([1/((k+1)**task_alpha) for k in range(self.num_task)])
    self.task_prob = task_prob/task_prob.sum()
    # task_vector
    self.task_for_task_vector = np.random.choice(self.num_task, self.num_seq, p=self.task_prob)
    self.task_ind = np.random.randint(0, self.num_labels, size=(self.num_task, self.num_classes))

  def get_seq(self):
    while True:
      if self.data_type=="bursty":
        if self.p_bursty > np.random.rand():
          # choise few shot tasks
          num_few_shot_task = self.num_seq//self.task_ways
          few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False, p=self.task_prob)
          tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
          
          # choise few shot items
          num_few_shot_class = self.num_seq//self.item_ways
          few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
          mus = self.mu[few_shot_class]
          mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
          
          # choice few shot labels
          labels = self.labels[few_shot_class]
          labels = np.repeat(labels, self.item_ways, axis=0) # expand ways
        
          # classes 
          classes = np.repeat(few_shot_class, self.item_ways)
          # add noise
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]
          task_ordering = np.random.permutation(self.num_seq)
          tasks = tasks[task_ordering]
          
          # task mapping
          labels = (labels + self.task_ind[tasks[0], classes].reshape(-1,1)) % self.num_labels
          
          # select query labels
          # query_class = np.random.choice(few_shot_class, 1)
          query_class = np.random.choice(self.num_classes, 1)
          query_task = np.random.choice(few_shot_task, 1)
          query_label = (self.labels[query_class] + self.task_ind[query_task, query_class]) % self.num_labels
          query_mu = self.mu[query_class]
          query_x = self.add_noise(query_mu)
          # concat
          x = torch.cat([x, query_x])
          labels = torch.cat([labels.flatten(), query_label.flatten()])
          tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(query_task).flatten()])
          
          yield {
              "tasks":tasks,
              "examples":x.to(torch.float32),
              "labels":labels,
              "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
          }
        
        else:
          # rank frequency
          num_few_shot_task = self.num_seq//self.task_ways
          few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False, p=self.task_prob)
          tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
          
          # choise few shot items
          num_few_shot_class = self.num_seq//self.item_ways
          few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
          mus = self.mu[few_shot_class]
          mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
          
          # choice few shot labels
          labels = self.labels[few_shot_class]
          labels = np.repeat(labels, self.item_ways, axis=0) # expand ways
        
          # classes 
          classes = np.repeat(few_shot_class, self.item_ways)
          # add noise
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]
          task_ordering = np.random.permutation(self.num_seq)
          tasks = tasks[task_ordering]
          
          labels = (labels + self.task_ind[tasks[0], classes].reshape(-1,1)) % self.num_labels
          # select query labels
          # query_class = np.random.choice(few_shot_class, 1)
          query_class = np.random.choice(classes, 1)
          query_task = np.random.choice(few_shot_task, 1)
          query_label = (self.labels[query_class] + self.task_ind[query_task, query_class]) % self.num_labels
          query_mu = self.mu[query_class]
          query_x = self.add_noise(query_mu)
          # concat
          x = torch.cat([x, query_x])
          labels = torch.cat([labels.flatten(), query_label.flatten()])
          tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(query_task).flatten()])
          
          yield {
              "tasks":tasks,
              "examples":x.to(torch.float32),
              "labels":labels,
              "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
          }
          
          
        """else:
          # rank frequency
          num_few_shot_task = self.num_seq//self.task_ways
          few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False, p=self.task_prob)
          tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
          
          classes = np.random.choice(self.num_classes, self.num_seq+1, p=self.prob)
          mus = self.mu[classes]
          labels = self.labels[classes]
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq+1)
          x = x[ordering]
          labels = labels[ordering]
          labels = (labels + tasks) % self.num_labels
          classes = classes[ordering]
          tasks = tasks[ordering]

          yield {
              "tasks":tasks,
              "examples":x.to(torch.float32),
              "labels":labels.flatten(),
              "classes" : torch.from_numpy(classes)
          }"""

      elif self.data_type == "no_support":
          num_few_shot_task = self.num_seq #//self.task_ways
          few_shot_task = np.random.choice(self.num_task, num_few_shot_task, p=self.task_prob)
          tasks = few_shot_task
          # tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
        
          # rank frequency
          classes = np.random.choice(self.num_classes, self.num_seq, p=self.prob)
          mus = self.mu[classes]
          # random label
          labels = np.random.randint(self.num_labels, size=(self.num_seq,1))
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]
          tasks = tasks[ordering]
          
          # select query labels
          query_class = np.random.choice(self.num_classes, 1)
          query_task = np.random.choice(few_shot_task, 1)
          query_label = self.labels[query_class]
          query_label = (query_label + self.task_ind[query_task, query_class]) % self.num_labels
          query_mu = self.mu[query_class]
          query_mu = self.add_noise(query_mu)
          
          # concat
          x = torch.cat([x, query_mu])
          labels = torch.cat([torch.from_numpy(labels).flatten(), query_label.flatten()])
          tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(query_task).flatten()])
          classes = np.concatenate([classes, query_class])
          
          # task vector example
          # choise few shot tasks
          tasks_for_task_vector = np.repeat(query_task, self.task_ways, axis=0).reshape(-1,1)
          
          # choise few shot items
          num_few_shot_class = self.num_seq//self.item_ways
          few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
          mus = self.mu[few_shot_class]
          mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
          
          # choice few shot labels
          labels_for_task_vector  = self.labels[few_shot_class]
          labels_for_task_vector  = np.repeat(labels_for_task_vector , self.item_ways, axis=0) # expand ways
        
          
          # classes 
          classes_for_task_vector  = np.repeat(few_shot_class, self.item_ways)
          # add noise
          x_for_task_vector = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq)
          x_for_task_vector = x_for_task_vector[ordering]
          labels_for_task_vector  = labels_for_task_vector[ordering]
          classes_for_task_vector = classes_for_task_vector[ordering]
          task_ordering = np.random.permutation(self.num_seq)
          tasks_for_task_vector = tasks_for_task_vector[task_ordering]
          
          labels_for_task_vector = (labels_for_task_vector  + tasks_for_task_vector) % self.num_labels
          
          # select query labels
          # query_class = np.random.choice(few_shot_class, 1)
          query_class = np.random.choice(self.num_classes, 1)
          # query_task = np.random.choice(few_shot_task, 1)
          query_label = (self.labels[query_class] + self.task_ind[query_task, query_class]) % self.num_labels
          query_mu = self.mu[query_class]
          query_x = self.add_noise(query_mu)
          # concat
          x_for_task_vector = torch.cat([x_for_task_vector, query_x])
          labels_for_task_vector = torch.cat([labels_for_task_vector.flatten(), query_label.flatten()])
          tasks_for_task_vector = torch.cat([torch.tensor(tasks_for_task_vector).flatten(), torch.tensor(query_task).flatten()])
      
          yield {
              "tasks": tasks,
              "examples":x.to(torch.float32),
              "labels":labels.flatten(),
              "classes" : torch.from_numpy(classes),
              # "task_vector" : {
              #     "tasks":tasks_for_task_vector,
              #     "examples":x_for_task_vector.to(torch.float32),
              #     "labels":labels_for_task_vector,
              #     "classes" : torch.cat([torch.from_numpy(classes_for_task_vector).flatten(), torch.from_numpy(query_class).flatten()])
              # }
          }
          
      elif self.data_type == "holdout":
                  # rank frequency
        num_few_shot_task = self.num_seq//self.task_ways
        few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False, p=self.task_prob)
        tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
        query_task = np.random.choice(few_shot_task, 1)
        tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(query_task).flatten()])
        
        # choise few shot example
        num_few_shot_class = self.num_seq//self.item_ways
        mus, labels = self._get_novel_class_seq(num_few_shot_class)
        # mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
        # labels = self.labels[few_shot_class]
        labels = np.repeat(labels, self.item_ways, axis=0) # expand ways
        classes = np.arange(num_few_shot_class)
        classes = np.repeat(classes, self.item_ways)
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        mus = mus[ordering]
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        # select query labels
        query_class_idx = np.random.choice(len(classes), 1)
        query_class = classes[query_class_idx]
        query_label = labels[query_class_idx]
        query_mu = mus[query_class_idx]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        
        yield {
            "tasks":tasks,
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }

      elif self.data_type == "flip":
        # choise few shot tasks
        num_few_shot_task = self.num_seq//self.task_ways
        few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False, p=self.task_prob)
        tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
        
        # choise few shot items
        num_few_shot_class = self.num_seq//self.item_ways
        few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
        mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
        
        # choice few shot labels
        labels = self.labels[few_shot_class]
        labels = np.repeat(labels, self.item_ways, axis=0) # expand ways
      
        # classes 
        classes = np.repeat(few_shot_class, self.item_ways)
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        task_ordering = np.random.permutation(self.num_seq)
        tasks = tasks[task_ordering]
        
        labels = (labels + self.task_ind[tasks[0], classes].reshape(-1,1)) % self.num_labels
        
        # select query labels
        # query_class = np.random.choice(few_shot_class, 1)
        query_class = np.random.choice(self.num_classes, 1)
        query_task = np.random.choice(few_shot_task, 1)
        query_label = (self.labels[query_class] + self.task_ind[query_task, query_class]) % self.num_labels
        query_mu = self.mu[query_class]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(query_task).flatten()])
        
        yield {
            "tasks":tasks,
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }

      elif self.data_type == "random_label":
        # choise few shot tasks
        num_few_shot_task = self.num_seq//self.task_ways
        few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False, p=self.task_prob)
        tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
        
        # choise few shot items
        num_few_shot_class = self.num_seq//self.item_ways
        few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
        mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
        
        # choice few shot labels
        labels = self.labels[few_shot_class]
        labels = np.repeat(labels, self.item_ways, axis=0) # expand ways
      
        # classes 
        classes = np.repeat(few_shot_class, self.item_ways)
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        task_ordering = np.random.permutation(self.num_seq)
        tasks = tasks[task_ordering]
        
        labels = (labels + self.task_ind[tasks[0], classes].reshape(-1,1)) % self.num_labels
        
        # random label
        label_ordering = np.random.permutation(self.num_seq)
        labels = labels[label_ordering]
        # to tensor
        # labels = torch.from_numpy(labels)
      
        
        # select query labels
        # query_class = np.random.choice(few_shot_class, 1)
        query_class = np.random.choice(self.num_classes, 1)
        query_task = np.random.choice(few_shot_task, 1)
        query_label = (self.labels[query_class] + self.task_ind[query_task, query_class]) % self.num_labels
        query_mu = self.mu[query_class]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(query_task).flatten()])
        
        yield {
            "tasks":tasks,
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }
        
  def get_seq_for_task_vector(self):
      while True:
        # choise few shot tasks
        # num_few_shot_task = self.num_seq//self.task_ways
        # few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False, p=self.task_prob)
        print(self.task_for_task_vector.shape)
        tasks = self.task_for_task_vector
        tasks = np.repeat(tasks, self.task_ways, axis=0).reshape(-1,1)
        
        # choise few shot items
        num_few_shot_class = self.num_seq//self.item_ways
        few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
        mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
        
        # choice few shot labels
        labels = self.labels[few_shot_class]
        labels = np.repeat(labels, self.item_ways, axis=0) # expand ways
      
        # classes 
        classes = np.repeat(few_shot_class, self.item_ways)
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        task_ordering = np.random.permutation(self.num_seq)
        tasks = tasks[task_ordering]
        
        labels = (labels + tasks) % self.num_labels
        
        # select query labels
        # query_class = np.random.choice(few_shot_class, 1)
        query_class = np.random.choice(self.num_classes, 1)
        query_task = np.random.choice(tasks, 1)
        query_label = (self.labels[query_class] + query_task) % self.num_labels
        query_mu = self.mu[query_class]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(query_task).flatten()])
        
        yield {
            "tasks":tasks,
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }
  

  def add_noise(self, x):
    x = (x+self.eps*torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(x.shape)))/(np.sqrt(1+self.eps**2))
    # x = (x+self.eps*np.random.normal(mean=0, std=np.sqrt(1/self.dim), size=(x.shape[0],1)))/(np.sqrt(1+self.eps**2))
    return x
  
  def _get_novel_class_seq(self,num_class):
    mu = torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(num_class,self.dim))
    labels = torch.randint(self.num_labels, size=(num_class,1))
    return mu, labels