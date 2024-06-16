import utils
from train_tools import mlm
import numpy as np

from models.create_diffusion import create_gaussian_diffusion
from diffusion_models.resample import create_named_schedule_sampler

def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mlm']:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['lu']:
        metric_logger.add_meter('loss_lu', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['unc']:
        metric_logger.add_meter('loss_unc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mim']:
        metric_logger.add_meter('loss_mim', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['sdm']:
        metric_logger.add_meter('loss_sdm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['id']:
        metric_logger.add_meter('loss_id', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['match']:
        metric_logger.add_meter('loss_match', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['diffusion']:
        metric_logger.add_meter('loss_diffusion', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['uncertainty']:
        metric_logger.add_meter('loss_irtr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['fine_grained']:
        metric_logger.add_meter('loss_fine_grained', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['gpt']:
        metric_logger.add_meter('loss_gpt_contrasive', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    if config['eda']:
        for i, (image, text, text_eda, idx, gpt) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
            text_input_eda = tokenizer(text_eda, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                       return_tensors="pt").to(device)

            text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator,
                                                        config)
            gpt_input = None
            diffusion = None
            schedule_sampler = None
            if config['gpt']:
                gpt_input = tokenizer(gpt, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
            if config['diffusion']:
                diffusion = create_gaussian_diffusion()
                schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
            add_loss= model(image, text_input.input_ids, text_input.attention_mask,
                                                text_ids_masked=text_ids_masked,
                                                masked_pos=masked_pos, masked_ids=masked_ids, idx=idx,
                                                text_ids_eda=text_input_eda.input_ids,
                                                text_atts_eda=text_input_eda.attention_mask,cur_epoch=epoch,
                                                diffusion=diffusion,
                                                schedule_sampler=schedule_sampler,
                                                gpt_input = gpt_input
                                                )
            loss = 0
            for n, v in add_loss.items():
                loss = loss + v

            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name)
            optimizer.step()
            scheduler.step()

            metric_logger.update(loss_itc=add_loss['loss_itc'].item())
            metric_logger.update(loss_itm=add_loss['loss_itm'].item())
            if config['mlm']:
                metric_logger.update(loss_mlm=add_loss['loss_mlm'].item())
            if config['lu']:
                metric_logger.update(loss_lu=add_loss['loss_lu'].item())
            if config['unc']:
                metric_logger.update(loss_unc=add_loss['loss_unc'].item())
            if config['mim']:
                metric_logger.update(loss_mim=add_loss['loss_mim'].item())
            if config['id']:
                metric_logger.update(loss_id=add_loss['loss_id'].item())
            if config['sdm']:
                metric_logger.update(loss_sdm=add_loss['loss_sdm'].item())
            if config['match']:
                metric_logger.update(loss_match=add_loss['loss_match'].item())
            if config['diffusion']:
                metric_logger.update(loss_diffusion=add_loss['loss_diffusion'].item())
            if config['uncertainty']:
                metric_logger.update(loss_irtr=add_loss['loss_irtr'].item())
            if config['fine_grained']:
                metric_logger.update(loss_fine_grained=add_loss['loss_fine_grained'].item())
            if config['gpt']:
                metric_logger.update(loss_gpt_contrasive=add_loss['loss_gpt_contrasive'].item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def train_attr(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mlm']:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_attr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mlm']:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['lu']:
        metric_logger.add_meter('loss_lu', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['unc']:
        metric_logger.add_meter('loss_unc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mim']:
        metric_logger.add_meter('loss_mim', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['sdm']:
        metric_logger.add_meter('loss_sdm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['id']:
        metric_logger.add_meter('loss_id', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['match']:
        metric_logger.add_meter('loss_match', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['diffusion']:
        metric_logger.add_meter('loss_diffusion', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['uncertainty']:
        metric_logger.add_meter('loss_irtr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['fine_grained']:
        metric_logger.add_meter('loss_fine_grained', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, text, idx, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        label = label.to(device, non_blocking=True)

        attr = ['the person is a woman', 'the person is a man',
                'the person is younger than 18 years old', 'the person is older than 18 years old',

                'the person with short hair', 'the person with long hair',
                'the person with a hat', 'the person without a hat',
                'the person with a backpack', 'the person without a backpack',
                'the person with a handbag', 'the person without a handbag',
                'the person with a bag', 'the person without a bag',

                'the person wears long sleeved upper clothes', 'the person wears short sleeved upper clothes',
                'the person wears long dress or long pants', 'the person wears short dress or short pants',
                'the person wears dress or skirt', 'the person wears pants or shorts',

                'the person wears black upper clothes', 'the person does not wear black upper clothes',
                'the person wears white upper clothes', 'the person does not wear white upper clothes',
                'the person wears red upper clothes', 'the person does not wear red upper clothes',
                'the person wears purple upper clothes', 'the person does not wear purple upper clothes',

                'the person wears yellow upper clothes', 'the person does not wear yellow upper clothes',
                'the person wears blue upper clothes', 'the person does not wear blue upper clothes',
                'the person wears green upper clothes', 'the person does not wear green upper clothes',
                'the person wears gray upper clothes', 'the person does not wear gray upper clothes',

                'the person wears black lower clothes', 'the person does not wear black lower clothes',
                'the person wears white lower clothes', 'the person does not wear white lower clothes',
                'the person wears purple lower clothes', 'the person does not wear purple lower clothes',
                'the person wears yellow lower clothes', 'the person does not wear yellow lower clothes',

                'the person wears blue lower clothes', 'the person does not wear blue lower clothes',
                'the person wears green lower clothes', 'the person does not wear green lower clothes',
                'the person wears pink lower clothes', 'the person does not wear pink lower clothes',
                'the person wears gray lower clothes', 'the person does not wear gray lower clothes',
                'the person wears brown lower clothes', 'the person does not wear brown lower clothes',

                ]
        attr_input = tokenizer(attr, padding='longest', max_length=config['max_tokens'],
                               return_tensors="pt").to(device)

        # mlm loss
        if config['mlm']:
            text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator,
                                                          config)
            attr_text_ids_masked, attr_masked_pos, attr_masked_ids = mlm(attr, attr_input, tokenizer, device,
                                                                         mask_generator, config,
                                                                         True)
            diffusion = None
            schedule_sampler = None
            if config['diffusion']:
                diffusion = create_gaussian_diffusion()
                schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
            add_loss = model(image, text_input.input_ids, text_input.attention_mask,
                                                            text_ids_masked=text_ids_masked, masked_pos=masked_pos,
                                                            masked_ids=masked_ids, idx=idx,
                                                            attr_text_ids=attr_input.input_ids,
                                                            attr_text_atts=attr_input.attention_mask,
                                                            attr_text_ids_masked=attr_text_ids_masked,
                                                            attr_masked_pos=attr_masked_pos,
                                                            attr_masked_ids=attr_masked_ids, label=label,
                                                            cur_epoch=epoch,
                                                            diffusion=diffusion, schedule_sampler=schedule_sampler)


        loss = 0
        for n, v in add_loss.items():
            loss = loss + v
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itc=add_loss['loss_itc'].item())
        metric_logger.update(loss_itm=add_loss['loss_itm'].item())
        metric_logger.update(loss_attr=add_loss['loss_attr'].item())
        if config['mlm']:
            metric_logger.update(loss_mlm=add_loss['loss_mlm'].item())
        if config['lu']:
            metric_logger.update(loss_lu=add_loss['loss_lu'].item())
        if config['unc']:
            metric_logger.update(loss_unc=add_loss['loss_unc'].item())
        if config['mim']:
            metric_logger.update(loss_mim=add_loss['loss_mim'].item())
        if config['id']:
            metric_logger.update(loss_id=add_loss['loss_id'].item())
        if config['sdm']:
            metric_logger.update(loss_sdm=add_loss['loss_sdm'].item())
        if config['match']:
            metric_logger.update(loss_match=add_loss['loss_match'].item())
        if config['diffusion']:
            metric_logger.update(loss_diffusion=add_loss['loss_diffusion'].item())
        if config['uncertainty']:
            metric_logger.update(loss_irtr=add_loss['loss_irtr'].item())
        if config['fine_grained']:
            metric_logger.update(loss_fine_grained=add_loss['loss_fine_grained'].item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}