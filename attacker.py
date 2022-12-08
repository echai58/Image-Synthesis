import torch

class Normalizer(torch.nn.Module):

    def __init__(self, new_mean, new_std):
        super(Normalizer, self).__init__()
        new_mean = new_mean[..., None, None]
        new_std = new_std[..., None, None]

        # needed to load in pretrained model correctly
        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        return (x - self.new_mean)/self.new_std

class Attacker(torch.nn.Module):
    def __init__(self, model, dataset):
        super(Attacker, self).__init__()
        self.normalize = Normalizer(dataset.mean, dataset.std)
        self.model = model
    
    def step(self, original, x, grad, eps, step_size):
        norms = []
        for sample in grad:
          norms.append(torch.norm(sample.flatten()))
        for i in range(len(grad)):
          norm = torch.clamp(norms[i],min=1e-12,max=float('inf'))
          grad[i] = grad[i]/norm
        x = x + grad * step_size

        # project step
        diff = (x - original).renorm(p=2, dim=0, maxnorm=eps)
        return torch.clamp(original + diff, 0, 1)
    
    def get_loss(self,x, target,normalize,loss_class):
        if normalize:
            x = self.normalize(x)
        return loss_class.loss(self.model,x,target)

    def forward(self, x, target, eps, step_size, iters, normalize, loss_class):
        original = x.detach().cuda()

        best_loss = [None, None] # loss, input

        for _ in range(iters):
            x = x.clone().detach().requires_grad_(True)
            loss = torch.mean(self.get_loss(x, target,normalize,loss_class))

            grad, = torch.autograd.grad(-loss, [x])

            with torch.no_grad():
                if best_loss[0] is None:
                    best_loss[0] = loss.clone().detach()
                    best_loss[1] = x.clone().detach()
                else:
                    index = -best_loss[0] < -loss
                    best_loss[0][index] = loss[index]
                    best_loss[1][index] = x[index].clone().detach()
                x = self.step(original, x, grad, eps, step_size)
                
        loss = torch.mean(self.get_loss(x, target,normalize,loss_class))
        index = -best_loss[0] < -loss
        best_loss[0][index] = loss[index]
        best_loss[1][index] = x[index].clone().detach()
        
        return best_loss[1]

class AttackerModel(torch.nn.Module):
    def __init__(self, model, dataset):
        super(AttackerModel, self).__init__()
        self.normalizer = Normalizer(dataset.mean, dataset.std)
        self.model = model
        self.attacker = Attacker(model, dataset)
        self.dataset = dataset

    def forward(self, input, target, eps, step_size, iters, normalize, loss_class):
        self.eval()
        adv_input = self.attacker(input, target, eps, step_size, iters, normalize, loss_class)
        normalized = self.normalizer(adv_input)
        output = self.model(normalized)
        return output, adv_input

