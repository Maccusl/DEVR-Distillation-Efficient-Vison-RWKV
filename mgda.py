from torch.autograd import Variable

from min_norm_solvers import gradient_normalizers,MinNormSolver


def mgda(optimizer, model, losses, loss_list= ["ofa", "infonce","kd", "gt"]):
    loss_data = {}
    grads = {}
    scale = {}
    optimizer.zero_grad()
    # count = {}
    for l in loss_list:
        optimizer.zero_grad()
        loss = losses[l]
        loss_data[l] = loss.item()
        loss.backward(retain_graph=True)
        grads[l] = []
        # count[l] = 0
        for param in model.parameters():
            if param.grad is not None:
                # count[l] += 1 
                grads[l].append(Variable(param.grad.data.clone(), requires_grad=False))
    # print(count)
    
    gn = gradient_normalizers(grads, loss_data, "loss+")
    for l in loss_list:
        for gr_i in range(len(grads[l])):
            grads[l][gr_i] = grads[l][gr_i] / gn[l]
            # print(grads[l][gr_i].shape)

    # Frank-Wolfe iteration to compute scales.
    # print(grads[l].shape for l in loss_list)
    sol, min_norm = MinNormSolver.find_min_norm_element([grads[l] for l in loss_list])
    for i, l in enumerate(loss_list):
        scale[l] = float(sol[i])
        
    
    return scale