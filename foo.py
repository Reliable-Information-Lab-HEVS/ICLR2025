import multiprocessing as mp
import time
import os
import torch
from concurrent.futures import ProcessPoolExecutor

from engine import loader
from helpers import utils


def dispatch_jobs(model_names, num_gpus, target_func, *func_args, **func_kwargs):
    """Run all jobs that need more than one gpu in parallel. Since the number of gpus needed by the models
    is variable, we cannot simply use a Pool of workers and map `target_func` to the Pool, or create processes and
    then ".join" them. To overcome this limitation, we use an infinite while loop that is refreshed by the main
    process every 10s. The dispatch of models to gpus is very naive: as soon as enough gpus are available to
    run the job that requires the less gpu, we launch it. Thus the gpu efficiency may not be the best possible.
    However, this would be extremely hard to improve on this simple strategy, especially since we do not know
    the runtime of each job.

    Parameters
    ----------
    model_names : _type_
        _description_
    num_gpus : _type_
        _description_
    """

    ctx = mp.get_context('spawn')

    # Retrieve the function that will be used (the function created by the decorator)
    # target_func_gpu_dispatch = globals()[f'{target_func.__name__}_gpu_dispatch']
    target_func_gpu_dispatch = target_func

    model_names = list(model_names)
    model_footprints = []

    # Estimate number of gpus needed for each model
    for model in model_names:
        quantization = model == 'bloom-176B'
        gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization)
        model_footprints.append(gpu_needed)

    print(model_footprints)

    # Sort both lists according to gpu footprint
    sorting = sorted(zip(model_names, model_footprints), key=lambda x: x[1])
    model_names = [x for x, _ in sorting]
    model_footprints = [x for _, x in sorting]

    # Initialize the lists we will maintain
    available_gpus = [i for i in range(num_gpus)]
    processes = []
    associated_gpus = []

    while True:

        no_sleep = False

        if len(available_gpus) >= model_footprints[0]:

            no_sleep = True

            # Remove them from the list of models to process
            name = model_names.pop(0)
            footprint = model_footprints.pop(0)

            # Update gpu resources
            allocated_gpus = available_gpus[0:footprint]
            available_gpus = available_gpus[footprint:]

            # p = ctx.Process(target=func_extended, args=(allocated_gpus, *func_args), kwargs=func_kwargs)
            p = ctx.Process(target=target_func_gpu_dispatch, args=func_args, kwargs=func_kwargs)
            # p = ctx.Process(target=target_func_on_gpu, args=(allocated_gpus, *func_args), kwargs=func_kwargs)
            p.start()

            # Add them to the list of running processes
            processes.append(p)
            associated_gpus.append(allocated_gpus)

        # Find the indices of the processes that are finished
        indices_to_remove = []
        for i, process in enumerate(processes):
            if not process.is_alive():
                indices_to_remove.append(i)
                # TODO: check this!!!
                process.close()

        # Update gpu resources
        released_gpus = [gpus for i, gpus in enumerate(associated_gpus) if i in indices_to_remove]
        available_gpus += [gpu for gpus in released_gpus for gpu in gpus]
        # Remove processes which are done
        processes = [process for i, process in enumerate(processes) if i not in indices_to_remove]
        associated_gpus = [gpus for i, gpus in enumerate(associated_gpus) if i not in indices_to_remove]

        # If we already scheduled all jobs, break from the infinite loop
        if len(model_names) == 0:
            break

        # Sleep for 10 seconds before restarting the loop and check if we have enough resources to launch
        # a new job
        if not no_sleep:
            time.sleep(10)

    # Sleep until all processes are finished (they have all been scheduled at this point)
    for process in processes:
        process.join()


@utils.duplicate_function_for_gpu_dispatch
def target(foo, bar):
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(f'Number of gpus seen by torch: {torch.cuda.device_count()}')
    time.sleep(5)
    print('Done!')


def target_func_on_gpu(visible_devices, *func_args, **func_kwargs):
        utils.set_cuda_visible_device(visible_devices)
        return target(*func_args, **func_kwargs)


LARGE_MODELS = (
    'gpt-neoX-20B',
    'llama2-70B',
)


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()

    model_footprints = []
    # Estimate number of gpus needed for each model
    for model in LARGE_MODELS:
        quantization = model == 'bloom-176B'
        gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization)
        model_footprints.append(gpu_needed)

    # utils.dispatch_jobs(LARGE_MODELS, model_footprints, num_gpus, target, [1,2], [3,4])
    dispatch_jobs(LARGE_MODELS, num_gpus, utils.target, [1,2], [3,4])
    # dispatch_jobs(LARGE_MODELS, num_gpus, target_func_on_gpu, [1,2], [3,4])

    # with ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp.get_context('spawn'),
    #                          initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
        
    #     _ = list(pool.map(target, LARGE_MODELS, LARGE_MODELS, chunksize=1))