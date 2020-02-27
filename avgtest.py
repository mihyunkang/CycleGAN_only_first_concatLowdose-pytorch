import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    print()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        fake_A_cpu = visuals['fake_A'][0][0].cpu()
        real_A_cpu = visuals['real_A'][0][0].cpu()
        psnr_1 = compare_psnr(fake_A_cpu.numpy(), real_A_cpu.numpy())
        ssim_1 = compare_ssim(fake_A_cpu.numpy(), real_A_cpu.numpy())
        print("psnr_1 {} ssim_1 {}".format(psnr_1, ssim_1))
        for j in range(400):
            model.test()           # run inference
            visuals['fake_A'] += model.get_current_visuals()['fake_A']  # get image results
        visuals['fake_A'] = visuals['fake_A']/400
        img_path = model.get_image_paths()     # get image paths
        fake_A_cpu = visuals['fake_A'][0][0].cpu()
        psnr_avg40 = compare_psnr(fake_A_cpu.numpy(), real_A_cpu.numpy())
        ssim_avg40 = compare_ssim(fake_A_cpu.numpy(), real_A_cpu.numpy())
        print("psnr_avg40 {} ssim_avg40 {}".format(psnr_avg40, ssim_avg40))
        if i % 5 == 0:  # save images to an HTML file
            print('processing avg 40 (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
