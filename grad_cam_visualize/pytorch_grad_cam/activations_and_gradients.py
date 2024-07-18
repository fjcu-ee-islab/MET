from torch import nn

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.reshape_transform_linear = nn.Linear(16,36)
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = activation.permute(1,2,0)
            target_channel = activation.shape[2]            
            self.reshape_transform_linear = nn.Linear(target_channel,36)
            activation = self.reshape_transform_linear(activation)
            activation = activation.permute(0,2,1)      
            activation = self.reshape_transform(activation)
            
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = grad.permute(1,2,0)
                target_channel = grad.shape[2]            
                self.reshape_transform_linear = nn.Linear(target_channel,36)
                grad = self.reshape_transform_linear(grad)
                grad = grad.permute(0,2,1)
                grad = self.reshape_transform(grad)

            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []

        polarity = x['kv']
        pixels = x['pixels']
        #embs, clf_logits = self.model(polarity, pixels)
        output = self.model(polarity, pixels)[0]

        return output
    
    '''
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)
    '''

    def release(self):
        for handle in self.handles:
            handle.remove()
