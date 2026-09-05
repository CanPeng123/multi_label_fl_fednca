import torch


def get_average_net(server, c_models, nums, device, check_l2_distance=False):
    """
    Performs federated averaging with equal averaging for 'clf_layer.etf_classifier.weight'.

    Args:
        server: the global model (with .state_dict())
        c_models: list of client model state_dicts
        nums: list of number of training samples per client
        device: torch.device
        check_l2_distance: bool, whether to print L2 distance diagnostics

    Returns:
        server: updated global model
    """
    total_num_train = sum(nums)
    total_l2_distance = 0.0
    num_models = len(c_models)

    with torch.no_grad():
        for key in server.comm_state_dict_names:
            if key == "clf_layer.etf_classifier.weight":
                # Equal averaging
                sum_param = torch.zeros_like(server.state_dict()[key]).float().to(device)
                count = 0
                for net in c_models:
                    if key in net:
                        sum_param += net[key].to(device)
                        count += 1
                if count > 0:
                    avg_param = sum_param / count
                    server.state_dict()[key].data.copy_(avg_param)
            else:
                # Data-size weighted averaging
                temp = torch.zeros_like(server.state_dict()[key]).float().to(device)
                for net, num in zip(c_models, nums):
                    if key in net:
                        temp += net[key].to(device) * (num / total_num_train)
                server.state_dict()[key].data.copy_(temp)

    # Calculate the L2 distance between each client model and the server model
    if check_l2_distance:
        for net in c_models:
            for key in server.comm_state_dict_names:
                if key not in net:
                    continue
                if not isinstance(net[key], torch.Tensor):
                    continue
                if net[key].shape != server.state_dict()[key].shape:
                    continue

                net_param = net[key].to(device)
                server_param = server.state_dict()[key].to(device)

                if not (torch.is_floating_point(net_param) and torch.is_floating_point(server_param)):
                    print(f"not floating point: {key}")
                    continue

                l2_distance = torch.norm(net_param - server_param, p=2)
                total_l2_distance += l2_distance.item()

        average_l2_distance = total_l2_distance / (num_models * len(server.comm_state_dict_names))
        print(f'Average L2 Distance between each model and the server model: {average_l2_distance:.4f}')

        total_norm = 0
        param_count = 0
        for param in server.parameters():
            if param.requires_grad and torch.is_floating_point(param):
                total_norm += param.norm(p=2).item()
                param_count += 1

        avg_weight_norm = total_norm / param_count
        print(f"Average weight L2 norm: {avg_weight_norm:.4f}")

    return server

