


@torch.no_grad()
def compare_iris_decoded(self, ckpt=None, train=False):
    # Sample n predictions per test data, choose the best among them for each metric

    calc_ssim = getattr(self.config.sampling, "ssim", False)
    calc_fvd = getattr(self.config.sampling, "fvd", False)

    # FVD
    if calc_fvd:

        if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
            calc_fvd1 = self.condf + self.config.sampling.num_frames_pred >= 9
            calc_fvd2 = calc_fvd3 = False
        elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation
            calc_fvd1 = self.condf + self.config.data.num_frames + self.futrf >= 9
            calc_fvd2 = calc_fvd3 = False
        elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred
            calc_fvd1 = self.condf + self.config.data.num_frames + self.futrf >= 9
            calc_fvd2 = self.condf + self.config.sampling.num_frames_pred >= 9
            calc_fvd3 = False
        elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen
            calc_fvd1 = calc_fvd3 = self.condf + self.config.sampling.num_frames_pred >= 9
            calc_fvd2 = False
        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:      # (1) Interp + (2) Pred + (3) Gen
            calc_fvd1 = self.condf + self.config.data.num_frames + self.futrf >= 9
            calc_fvd2 = calc_fvd3 = self.condf + self.config.sampling.num_frames_pred >= 9
        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:          # (1) Interp + (3) Gen
            calc_fvd1 = self.condf + self.config.data.num_frames + self.futrf >= 9
            calc_fvd2 = False
            calc_fvd3 = self.condf + self.config.sampling.num_frames_pred >= 9

        if calc_fvd1 or calc_fvd2 or calc_fvd3:
            i3d = load_i3d_pretrained(self.config.device)

        self.calc_fvd1, self.calc_fvd2, self.calc_fvd3 = calc_fvd1, calc_fvd2, calc_fvd3

    else:
        self.calc_fvd1, self.calc_fvd2, self.calc_fvd3 = calc_fvd1, calc_fvd2, calc_fvd3 = False, False, False

        if calc_ssim is False:
            return {}

    self.start_time = time.time()
    max_data_iter = self.config.sampling.max_data_iter
    preds_per_test = getattr(self.config.sampling, 'preds_per_test', 1)

    # Conditional
    conditional = self.config.data.num_frames_cond > 0
    assert conditional, f"Video generating model has to be conditional! num_frames_cond has to be > 0! Given {self.config.data.num_frames_cond}"
    cond = None
    prob_mask_cond = getattr(self.config.data, 'prob_mask_cond', 0.0)

    # Future
    future = getattr(self.config.data, "num_frames_future", 0)
    prob_mask_future = getattr(self.config.data, 'prob_mask_future', 0.0)

    # Collate fn for n repeats
    def my_collate(batch):
        data, embedding = zip(*batch)
        data = torch.stack(data).repeat_interleave(preds_per_test, dim=0)
        embedding = torch.stack(embedding).repeat_interleave(preds_per_test, dim=0)
        return data, embedding

    # Data
    if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
        num_frames_pred = self.config.sampling.num_frames_pred
    elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation
        num_frames_pred = self.config.data.num_frames
    elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred
        num_frames_pred = max(self.config.data.num_frames, self.config.sampling.num_frames_pred)
    elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen
        num_frames_pred = self.config.sampling.num_frames_pred
    elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred + (3) Gen
        num_frames_pred = self.config.sampling.num_frames_pred
    elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:     # (1) Interp + (3) Gen
        num_frames_pred = max(self.config.data.num_frames, self.config.sampling.num_frames_pred)
    
    """
    dataset_train, dataset_test = get_dataset(self.args.data_path, self.config, video_frames_pred=num_frames_pred, start_at=self.args.start_at)
    dataset = dataset_train if getattr(self.config.sampling, "train", False) else dataset_test
    dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size//preds_per_test, shuffle=True,
                            num_workers=self.config.data.num_workers, drop_last=False, collate_fn=my_collate)
    data_iter = iter(dataloader)
    """
    # num_frames_pred is 28!!
    assert self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync # Always (1) Interp + (2) Pred + (3) Gen
    total_frames = self.condf + num_frames_pred + self.futrf
    dataset = AtariDataset2(self.args.data_path, total_frames, transforms=get_atari_transform(self.config.data.image_size), episode_end_frames=self.config.episode_end)
    dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size//preds_per_test, shuffle=True,
                            num_workers=self.config.data.num_workers, drop_last=False, collate_fn=my_collate)
    data_iter = iter(dataloader)

    if self.config.sampling.data_init:
        dataloader2 = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                    num_workers=self.config.data.num_workers, drop_last=False)
        data_iter2 = iter(dataloader2)

    vid_mse, vid_ssim, vid_lpips = [], [], []
    vid_mse2, vid_ssim2, vid_lpips2 = [], [], []
    real_embeddings, real_embeddings2, real_embeddings_uncond = [], [], []
    fake_embeddings, fake_embeddings2, fake_embeddings_uncond = [], [], []

    T2 = Transforms.Compose([Transforms.Resize((128, 128)),
                    Transforms.ToTensor(),
                    Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                        std=(0.5, 0.5, 0.5))])
    model_lpips = eval_models.PerceptualLoss(model='net-lin',net='alex', device=self.config.device) # already in test mode and dataparallel
    #model_lpips = torch.nn.DataParallel(model_lpips)
    #model_lpips.eval()
    # Sampler
    sampler = self.get_sampler()

    for i, (real_, real_2) in tqdm(enumerate(dataloader), total=min(max_data_iter, len(dataloader)), desc="\nvideo_gen dataloader"):

        if i >= max_data_iter: # stop early
            break

        real_ = data_transform(self.config, real_)
        real_2 = data_transform(self.config, real_2)
        
        _, cond_original, _ = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred,
                                                prob_mask_cond=0.0, prob_mask_future=0.0, conditional=conditional)

        cond_original = inverse_data_transform(self.config, cond_original)


        # (1) Conditional Video Predition/Interpolation : Calc MSE,etc. and FVD on fully cond model i.e. prob_mask_cond=0.0
        # This is prediction if future = 0, else this is interpolation

        logging.info(f"(1) Video {'Pred' if future == 0 else 'Interp'}")

        # Video Prediction
        if future == 0:
            num_frames_pred = self.config.sampling.num_frames_pred
            logging.info(f"PREDICTING {num_frames_pred} frames, using a {self.config.data.num_frames} frame model conditioned on {self.config.data.num_frames_cond} frames, subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")
        # Video Interpolation
        else:
            num_frames_pred = self.config.data.num_frames
            logging.info(f"INTERPOLATING {num_frames_pred} frames, using a {self.config.data.num_frames} frame model conditioned on {self.config.data.num_frames_cond} cond + {future} future frames, subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")

        real, cond, cond_mask = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred,
                                                prob_mask_cond=0.0, prob_mask_future=0.0, conditional=conditional)
        real = inverse_data_transform(self.config, real)
        cond = cond.to(self.config.device)

        pred, _, _ = conditioning_fn(self.config, real_2, num_frames_pred=num_frames_pred,
                                                prob_mask_cond=0.0, prob_mask_future=0.0, conditional=conditional)
        
        pred = inverse_data_transform(self.config, pred)

        if real.shape[1] < pred.shape[1]: # We cannot calculate MSE, PSNR, SSIM
            print("-------- Warning: Cannot calculate metrics because predicting beyond the training data range --------")
            for ii in range(len(pred)):
                vid_mse.append(0)
                vid_ssim.append(0)
                vid_lpips.append(0)
        else:
            # Calculate MSE, PSNR, SSIM
            for ii in range(len(pred)):
                mse, avg_ssim, avg_distance = 0, 0, 0
                for jj in range(num_frames_pred):

                    # MSE (and PSNR)
                    pred_ij = pred[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                    real_ij = real[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                    mse += F.mse_loss(real_ij, pred_ij)

                    pred_ij_pil = Transforms.ToPILImage()(pred_ij).convert("RGB")
                    real_ij_pil = Transforms.ToPILImage()(real_ij).convert("RGB")

                    # SSIM
                    pred_ij_np_grey = np.asarray(pred_ij_pil.convert('L'))
                    real_ij_np_grey = np.asarray(real_ij_pil.convert('L'))
                    if self.config.data.dataset.upper() == "STOCHASTICMOVINGMNIST" or self.config.data.dataset.upper() == "MOVINGMNIST":
                        # ssim is the only metric extremely sensitive to gray being compared to b/w 
                        pred_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(pred_ij)).convert("RGB").convert('L'))
                        real_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(real_ij)).convert("RGB").convert('L'))
                    avg_ssim += ssim(pred_ij_np_grey, real_ij_np_grey, data_range=255, gaussian_weights=True, use_sample_covariance=False)

                    # Calculate LPIPS
                    pred_ij_LPIPS = T2(pred_ij_pil).unsqueeze(0).to(self.config.device)
                    real_ij_LPIPS = T2(real_ij_pil).unsqueeze(0).to(self.config.device)
                    avg_distance += model_lpips.forward(real_ij_LPIPS, pred_ij_LPIPS)

                vid_mse.append(mse / num_frames_pred)
                vid_ssim.append(avg_ssim / num_frames_pred)
                vid_lpips.append(avg_distance.data.item() / num_frames_pred)


        # (2) Conditional Video Predition, if (1) was Interpolation : Calc MSE,etc. and FVD on fully cond model i.e. prob_mask_cond=0.0
        # unless prob_mask_sync is True, in which case perform (3) uncond gen

        second_calc = False
        if future > 0 and prob_mask_future > 0.0 and not self.prob_mask_sync:

            second_calc = True
            logging.info(f"(2) Video Pred")

            num_frames_pred = self.config.sampling.num_frames_pred

            logging.info(f"PREDICTING {num_frames_pred} frames, using a {self.config.data.num_frames} frame model conditioned on {self.config.data.num_frames_cond} frames, subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")
            #print("real")
            #print(real_.shape)
            _, cond_original2, _ = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred,
                                                        prob_mask_cond=0.0, prob_mask_future=1.0, conditional=conditional)
            cond_original2 = inverse_data_transform(self.config, cond_original2)

            real2, cond, cond_mask = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred,
                                                        prob_mask_cond=0.0, prob_mask_future=1.0, conditional=conditional)
            #print(real2.shape)
            real2 = inverse_data_transform(self.config, real2)
            cond = cond.to(self.config.device)

            pred2, _, _ = conditioning_fn(self.config, real_2, num_frames_pred=num_frames_pred,
                                                        prob_mask_cond=0.0, prob_mask_future=1.0, conditional=conditional)
            pred2 = inverse_data_transform(self.config, pred2)
            # pred has length of multiple of n (because we repeat data sample n times)

            if real2.shape[1] < pred2.shape[1]: # We cannot calculate MSE, PSNR, SSIM
                print("-------- Warning: Cannot calculate metrics because predicting beyond the training data range --------")
                for ii in range(len(pred)):
                    vid_mse.append(0)
                    vid_ssim.append(0)
                    vid_lpips.append(0)
            else:
                ###
                #print(pred2.shape)
                #print(real2.shape)
                ###
                # Calculate MSE, PSNR, SSIM
                for ii in range(len(pred2)):
                    mse, avg_ssim, avg_distance = 0, 0, 0
                    for jj in range(num_frames_pred):

                        # MSE (and PSNR)
                        pred_ij = pred2[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                        real_ij = real2[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                        mse += F.mse_loss(real_ij, pred_ij)

                        pred_ij_pil = Transforms.ToPILImage()(pred_ij).convert("RGB")
                        real_ij_pil = Transforms.ToPILImage()(real_ij).convert("RGB")

                        # SSIM
                        pred_ij_np_grey = np.asarray(pred_ij_pil.convert('L'))
                        real_ij_np_grey = np.asarray(real_ij_pil.convert('L'))
                        if self.config.data.dataset.upper() == "STOCHASTICMOVINGMNIST" or self.config.data.dataset.upper() == "MOVINGMNIST":
                            # ssim is the only metric extremely sensitive to gray being compared to b/w 
                            pred_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(pred_ij)).convert("RGB").convert('L'))
                            real_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(real_ij)).convert("RGB").convert('L'))
                        avg_ssim += ssim(pred_ij_np_grey, real_ij_np_grey, data_range=255, gaussian_weights=True, use_sample_covariance=False)

                        # Calculate LPIPS
                        pred_ij_LPIPS = T2(pred_ij_pil).unsqueeze(0).to(self.config.device)
                        real_ij_LPIPS = T2(real_ij_pil).unsqueeze(0).to(self.config.device)
                        avg_distance += model_lpips.forward(real_ij_LPIPS, pred_ij_LPIPS)

                    vid_mse2.append(mse / num_frames_pred)
                    vid_ssim2.append(avg_ssim / num_frames_pred)
                    vid_lpips2.append(avg_distance.data.item() / num_frames_pred)

        # FVD

        logging.info(f"fvd1 {calc_fvd1}, fvd2 {calc_fvd2}, fvd3 {calc_fvd3}")
        pred_uncond = None
        if calc_fvd1 or calc_fvd2 or calc_fvd3:

            # (3) Unconditional Video Generation: We must redo the predictions with no input conditioning for unconditional FVD
            if calc_fvd3:

                logging.info(f"(3) Video Gen - Uncond - FVD")

                # If future = 0, we must make more since first ones are empty frames!
                # Else, run only 1 iteration, and make only num_frames
                num_frames_pred = self.config.data.num_frames_cond + self.config.sampling.num_frames_pred

                logging.info(f"GENERATING (Uncond) {num_frames_pred} frames, using a {self.config.data.num_frames} frame model (conditioned on {self.config.data.num_frames_cond} cond + {self.config.data.num_frames_future} futr frames), subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")

                # We mask cond
                _, cond_fvd, cond_mask_fvd = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred,
                                                                prob_mask_cond=1.0, prob_mask_future=1.0, conditional=conditional, encodings=encodings_ if self.config.use_encoding else None)
                cond_fvd = cond_fvd.to(self.config.device)

                # z
                init_samples_shape = (cond_fvd.shape[0], self.config.data.channels*self.config.data.num_frames,
                                        self.config.data.image_size, self.config.data.image_size)
                if self.version == "SMLD":
                    z = torch.rand(init_samples_shape, device=self.config.device)
                    z = data_transform(self.config, z)
                elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                    if getattr(self.config.model, 'gamma', False):
                        used_k, used_theta = net.k_cum[0], net.theta_t[0]
                        z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                        z = z - used_k*used_theta
                    else:
                        z = torch.randn(init_samples_shape, device=self.config.device)
                        # z = data_transform(self.config, z)

                # init_samples
                if self.config.sampling.data_init:
                    try:
                        real_init, _ = next(data_iter2)
                    except StopIteration:
                        if self.config.data.dataset.upper() == 'FFHQ':
                            dataloader2 = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                        data_iter2 = iter(dataloader2)
                        real_init, _ = next(data_iter2)
                    real_init = data_transform(self.config, real_init)
                    real_init, _, _ = conditioning_fn(self.config, real_init, conditional=conditional)
                    real_init = real_init.to(self.config.device)
                    real_init1 = real_init[:, :self.config.data.channels*self.config.data.num_frames]
                    if self.version == "SMLD":
                        z = torch.randn_like(real_init1)
                        init_samples = real_init1 + float(self.config.model.sigma_begin) * z
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        alpha = net.alphas[0]
                        z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                        init_samples = alpha.sqrt() * real_init1 + (1 - alpha).sqrt() * z
                else:
                    init_samples = z

                if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                    n_iter_frames = (num_frames_pred)
                else:
                    n_iter_frames = ceil(num_frames_pred / self.config.data.num_frames)

                pred_samples = []

                if self.config.use_encoding:
                    split_index = self.config.data.channels*(self.config.data.num_frames_cond+self.config.data.num_frames_future)
                    cond_fvd, cond_emb = torch.split(cond_fvd, [split_index, cond_fvd.shape[1]-split_index], dim=1)

                for i_frame in tqdm(range(n_iter_frames), desc="Generating video frames"):
                    if self.config.use_encoding:
                        if self.config.encoding_type == "bilinear":
                            final_cond = torch.cat([cond_fvd, cond_emb[:,i_frame*self.config.data.num_frames*3:(i_frame+1)*self.config.data.num_frames*3]], dim=1)
                            if final_cond.shape[1] < cond_fvd.shape[1] + self.config.data.num_frames*3:
                                num_zeros = cond_fvd.shape[1] + self.config.data.num_frames*3 - final_cond.shape[1]
                                final_cond = torch.cat([final_cond, torch.zeros([final_cond.shape[0], num_zeros, *final_cond.shape[2:]], device=self.config.device)], dim=1)
                        else:
                            final_cond = torch.cat([cond_fvd, cond_emb[:,i_frame*self.config.data.num_frames:(i_frame+1)*self.config.data.num_frames]], dim=1)
                            if final_cond.shape[1] < cond_fvd.shape[1] + self.config.data.num_frames:
                                num_zeros = cond_fvd.shape[1] + self.config.data.num_frames - final_cond.shape[1]
                                final_cond = torch.cat([final_cond, torch.zeros([final_cond.shape[0], num_zeros, *final_cond.shape[2:]], device=self.config.device)], dim=1)
                    else:
                        final_cond = cond_fvd

                    # Generate samples
                    gen_samples = sampler(init_samples if i_frame==0 or getattr(self.config.sampling, 'init_prev_t', -1) <= 0 else gen_samples,
                                            scorenet, cond=final_cond, cond_mask=cond_mask_fvd,
                                            n_steps_each=self.config.sampling.n_steps_each, step_lr=self.config.sampling.step_lr,
                                            verbose=True if not train else False, final_only=True, denoise=self.config.sampling.denoise,
                                            subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                            clip_before=getattr(self.config.sampling, 'clip_before', True),
                                            t_min=getattr(self.config.sampling, 'init_prev_t', -1), log=True if not train else False,
                                            gamma=getattr(self.config.model, 'gamma', False))
                    gen_samples = gen_samples[-1].reshape(gen_samples[-1].shape[0], self.config.data.channels*self.config.data.num_frames,
                                                            self.config.data.image_size, self.config.data.image_size)
                    pred_samples.append(gen_samples.to('cpu'))

                    if i_frame == n_iter_frames - 1:
                        continue

                    # cond -> [cond[n:], real[:n]]
                    if future == 0:
                        if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                            cond_fvd = torch.cat([cond_fvd[:, self.config.data.channels:], gen_samples[:, :self.config.data.channels]], dim=1)
                        else:
                            cond_fvd = torch.cat([cond_fvd[:, self.config.data.channels*self.config.data.num_frames:],
                                                    gen_samples[:, self.config.data.channels*max(0, self.config.data.num_frames - self.config.data.num_frames_cond):]
                                                    ], dim=1)
                    else:
                        if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                            cond_fvd = torch.cat([cond_fvd[:, self.config.data.channels:],
                                                    gen_samples[:, :self.config.data.channels],
                                                    cond_fvd[:, -self.config.data.channels*future:]   # future frames are always there, but always 0
                                                    ], dim=1)
                        else:
                            cond_fvd = torch.cat([cond_fvd[:, self.config.data.channels*self.config.data.num_frames:-self.config.data.channels*future],
                                                    gen_samples[:, self.config.data.channels*max(0, self.config.data.num_frames - self.config.data.num_frames_cond):],
                                                    cond_fvd[:, -self.config.data.channels*future:]   # future frames are always there, but always 0
                                                    ], dim=1)

                    # Make cond_mask one
                    if i_frame == 0:
                        cond_mask_fvd = cond_mask_fvd.fill_(1)
                    # resample new random init
                    if self.version == "SMLD":
                        z = torch.rand(init_samples_shape, device=self.config.device)
                        z = data_transform(self.config, z)
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        if getattr(self.config.model, 'gamma', False):
                            used_k, used_theta = net.k_cum[0], net.theta_t[0]
                            z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                            z = z - used_k*used_theta
                        else:
                            z = torch.randn(init_samples_shape, device=self.config.device)

                    # init_samples
                    if self.config.sampling.data_init:
                        if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                            real_init1 = real_init[self.config.data.channels*(i_frame+1):self.config.data.channels*(i_frame+1+self.config.data.num_frames)]
                        else:
                            real_init1 = real_init[(i_frame+1)*self.config.data.channels*self.config.data.num_frames:(i_frame+2)*self.config.data.channels*self.config.data.num_frames]
                        if self.version == "SMLD":
                            z = torch.randn_like(real_init1)
                            init_samples = real_init1 + float(self.config.model.sigma_begin) * z
                        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                            alpha = net.alphas[0]
                            z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                            init_samples = alpha.sqrt() * real_init1 + (1 - alpha).sqrt() * z
                    else:
                        init_samples = z

                pred_uncond = torch.cat(pred_samples, dim=1)[:, :self.config.data.channels*num_frames_pred]
                pred_uncond = inverse_data_transform(self.config, pred_uncond)

            def to_i3d(x):
                x = x.reshape(x.shape[0], -1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                if self.config.data.channels == 1:
                    x = x.repeat(1, 1, 3, 1, 1) # hack for greyscale images
                x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
                return x

            if (calc_fvd1 or (calc_fvd3 and not second_calc)) and real.shape[1] >= pred.shape[1]:

                # real
                if future == 0:
                    real_fvd = torch.cat([
                        cond_original[:, :self.config.data.num_frames_cond*self.config.data.channels],
                        real
                    ], dim=1)[::preds_per_test]    # Ignore the repeated ones
                else:
                    real_fvd = torch.cat([
                        cond_original[:, :self.config.data.num_frames_cond*self.config.data.channels],
                        real,
                        cond_original[:, -future*self.config.data.channels:]
                    ], dim=1)[::preds_per_test]    # Ignore the repeated ones
                real_fvd = to_i3d(real_fvd)
                real_embeddings.append(get_fvd_feats(real_fvd, i3d=i3d, device=self.config.device))

                # fake
                if future == 0:
                    fake_fvd = torch.cat([
                        cond_original[:, :self.config.data.num_frames_cond*self.config.data.channels], pred], dim=1)
                else:
                    fake_fvd = torch.cat([
                        cond_original[:, :self.config.data.num_frames_cond*self.config.data.channels],
                        pred,
                        cond_original[:, -future*self.config.data.channels:]
                    ], dim=1)
                fake_fvd = to_i3d(fake_fvd)
                fake_embeddings.append(get_fvd_feats(fake_fvd, i3d=i3d, device=self.config.device))

            # fake2 : fvd_cond if fvd was fvd_interp
            if (second_calc and (calc_fvd2 or calc_fvd3)) and real.shape[1] >= pred.shape[1]: # only cond, but real has all frames req for interp

                # real2
                real_fvd2 = torch.cat([
                    cond_original2[:, :self.config.data.num_frames_cond*self.config.data.channels],
                    real2
                ], dim=1)[::preds_per_test]    # Ignore the repeated ones
                real_fvd2 = to_i3d(real_fvd2)
                real_embeddings2.append(get_fvd_feats(real_fvd2, i3d=i3d, device=self.config.device))

                # fake2
                fake_fvd2 = torch.cat([
                    cond_original2[:, :self.config.data.num_frames_cond*self.config.data.channels],
                    pred2
                ], dim=1)
                fake_fvd2 = to_i3d(fake_fvd2)
                fake_embeddings2.append(get_fvd_feats(fake_fvd2, i3d=i3d, device=self.config.device))

            # (3) fake 3: uncond
            if calc_fvd3:
                # real uncond
                real_embeddings_uncond.append(real_embeddings2[-1] if second_calc else real_embeddings[-1])

                # fake uncond
                fake_fvd_uncond = torch.cat([pred_uncond], dim=1) # We don't want to input the zero-mask
                fake_fvd_uncond = to_i3d(fake_fvd_uncond)
                fake_embeddings_uncond.append(get_fvd_feats(fake_fvd_uncond, i3d=i3d, device=self.config.device))

        if i == 0 or preds_per_test == 1: # Save first mini-batch or save them all
            cond = cond_original

            no_metrics = False
            if real.shape[1] < pred.shape[1]: # Pad with zeros to prevent bugs
                no_metrics = True
                real = torch.cat([real, torch.zeros(real.shape[0], pred.shape[1]-real.shape[1], real.shape[2], real.shape[3])], dim=1)

            if future > 0:
                cond, futr = torch.tensor_split(cond, (self.config.data.num_frames_cond*self.config.data.channels,), dim=1)

            # Save gif
            gif_frames_cond = []
            gif_frames_pred, gif_frames_pred2, gif_frames_pred3 = [], [], []
            gif_frames_futr = []

            # cond : # we show conditional frames, and real&pred side-by-side
            for t in range(cond.shape[1]//self.config.data.channels):
                cond_t = cond[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                frame = torch.cat([cond_t, 0.5*torch.ones(*cond_t.shape[:-1], 2), cond_t], dim=-1)
                frame = frame.permute(0, 2, 3, 1).numpy()
                frame = np.stack([putText(f.copy(), f"{t+1:2d}p", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                nrow = ceil(np.sqrt(2*cond.shape[0])/2)
                gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                gif_frames_cond.append((gif_frame*255).astype('uint8'))
                if t == 0:
                    gif_frames_cond.append((gif_frame*255).astype('uint8'))
                del frame, gif_frame

            # pred
            for t in range(pred.shape[1]//self.config.data.channels):
                real_t = real[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                pred_t = pred[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                frame = torch.cat([real_t, 0.5*torch.ones(*pred_t.shape[:-1], 2), pred_t], dim=-1)
                frame = frame.permute(0, 2, 3, 1).numpy()
                frame = np.stack([putText(f.copy(), f"{t+1:02d}", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                nrow = ceil(np.sqrt(2*pred.shape[0])/2)
                gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                gif_frames_pred.append((gif_frame*255).astype('uint8'))
                if t == pred.shape[1]//self.config.data.channels - 1 and future == 0:
                    gif_frames_pred.append((gif_frame*255).astype('uint8'))
                del frame, gif_frame

            # pred2
            if second_calc:
                #print(pred2.shape)
                #print(real2.shape)
                for t in range(pred2.shape[1]//self.config.data.channels):
                    real_t = real2[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                    # print(real_t.shape)
                    pred_t = pred2[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                    # print(pred_t.shape)
                    frame = torch.cat([real_t, 0.5*torch.ones(*pred_t.shape[:-1], 2), pred_t], dim=-1)
                    frame = frame.permute(0, 2, 3, 1).numpy()
                    frame = np.stack([putText(f.copy(), f"{t+1:02d}", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                    nrow = ceil(np.sqrt(2*pred.shape[0])/2)
                    gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                    gif_frames_pred2.append((gif_frame*255).astype('uint8'))
                    if t == pred2.shape[1]//self.config.data.channels - 1:
                        gif_frames_pred2.append((gif_frame*255).astype('uint8'))
                    del frame, gif_frame

            # pred_uncond
            if pred_uncond is not None:
                for t in range(pred_uncond.shape[1]//self.config.data.channels):
                    frame = pred_uncond[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                    frame = frame.permute(0, 2, 3, 1).numpy()
                    frame = np.stack([putText(f.copy(), f"{t+1:02d}", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                    nrow = ceil(np.sqrt(2*pred_uncond.shape[0])/2)
                    gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                    gif_frames_pred3.append((gif_frame*255).astype('uint8'))
                    if t == pred_uncond.shape[1]//self.config.data.channels - 1:
                        gif_frames_pred3.append((gif_frame*255).astype('uint8'))
                    del frame, gif_frame

            # futr
            if future > 0: # if conditional, we show conditional frames, and real&pred, and future frames side-by-side
                for t in range(futr.shape[1]//self.config.data.channels):
                    futr_t = futr[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                    frame = torch.cat([futr_t, 0.5*torch.ones(*futr_t.shape[:-1], 2), futr_t], dim=-1)
                    frame = frame.permute(0, 2, 3, 1).numpy()
                    frame = np.stack([putText(f.copy(), f"{t+1:2d}f", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                    nrow = ceil(np.sqrt(2*futr.shape[0])/2)
                    gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                    gif_frames_futr.append((gif_frame*255).astype('uint8'))
                    if t == futr.shape[1]//self.config.data.channels - 1:
                        gif_frames_futr.append((gif_frame*255).astype('uint8'))
                    del frame, gif_frame

            # Save gif
            if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
                imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_pred_{ckpt}_{i}.gif"),
                                    [*gif_frames_cond, *gif_frames_pred], fps=4)
            elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation
                imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_interp_{ckpt}_{i}.gif"),
                                    [*gif_frames_cond, *gif_frames_pred, *gif_frames_futr], fps=4)
            elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred
                imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_interp_{ckpt}_{i}.gif"),
                                    [*gif_frames_cond, *gif_frames_pred, *gif_frames_futr], fps=4)
                imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_pred_{ckpt}_{i}.gif"),
                                    [*gif_frames_cond, *gif_frames_pred2], fps=4)
            elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen
                imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_pred_{ckpt}_{i}.gif"),
                                    [*gif_frames_cond, *gif_frames_pred], fps=4)
                if len(gif_frames_pred3) > 0:
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_gen_{ckpt}_{i}.gif"),
                                        gif_frames_pred3, fps=4)
            elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:     # (1) Interp + (2) Pred + (3) Gen
                imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_interp_{ckpt}_{i}.gif"),
                                    [*gif_frames_cond, *gif_frames_pred, *gif_frames_futr], fps=4)
                imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_pred_{ckpt}_{i}.gif"),
                                    [*gif_frames_cond, *gif_frames_pred2], fps=4)
                if len(gif_frames_pred3) > 0:
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_gen_{ckpt}_{i}.gif"),
                                        gif_frames_pred3, fps=4)
            elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:     # (1) Interp + (3) Gen
                imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_interp_{ckpt}_{i}.gif"),
                                    [*gif_frames_cond, *gif_frames_pred, *gif_frames_futr], fps=4)
                if len(gif_frames_pred3) > 0:
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_gen_{ckpt}_{i}.gif"),
                                        gif_frames_pred3, fps=4)

            del gif_frames_cond, gif_frames_pred, gif_frames_pred2, gif_frames_pred3, gif_frames_futr

            # Stretch out multiple frames horizontally

            def save_pred(pred, real):
                if train:
                    torch.save({"cond": cond, "pred": pred, "real": real},
                                os.path.join(self.args.log_sample_path, f"videos_pred_{ckpt}.pt"))
                else:
                    torch.save({"cond": cond, "pred": pred, "real": real},
                                os.path.join(self.args.video_folder, f"videos_pred_{ckpt}.pt"))
                cond_im = stretch_image(cond, self.config.data.channels, self.config.data.image_size)
                pred_im = stretch_image(pred, self.config.data.channels, self.config.data.image_size)
                real_im = stretch_image(real, self.config.data.channels, self.config.data.image_size)
                padding_hor = 0.5*torch.ones(*real_im.shape[:-1], 2)
                real_data = torch.cat([cond_im, padding_hor, real_im], dim=-1)
                pred_data = torch.cat([0.5*torch.ones_like(cond_im), padding_hor, pred_im], dim=-1)
                padding_ver = 0.5*torch.ones(*real_im.shape[:-2], 2, real_data.shape[-1])
                data = torch.cat([real_data, padding_ver, pred_data], dim=-2)
                # Save
                nrow = ceil(np.sqrt((self.config.data.num_frames_cond+self.config.sampling.num_frames_pred)*pred.shape[0])/(self.config.data.num_frames_cond+self.config.sampling.num_frames_pred))
                image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
                if train:
                    save_image(image_grid, os.path.join(self.args.log_sample_path, f"videos_stretch_pred_{ckpt}_{i}.png"))
                else:
                    save_image(image_grid, os.path.join(self.args.video_folder, f"videos_stretch_pred_{ckpt}_{i}.png"))

            def save_interp(pred, real):
                if train:
                    torch.save({"cond": cond, "pred": pred, "real": real, "futr": futr},
                                os.path.join(self.args.log_sample_path, f"videos_interp_{ckpt}.pt"))
                else:
                    torch.save({"cond": cond, "pred": pred, "real": real, "futr": futr},
                                os.path.join(self.args.video_folder, f"videos_interp_{ckpt}.pt"))
                cond_im = stretch_image(cond, self.config.data.channels, self.config.data.image_size)
                pred_im = stretch_image(pred, self.config.data.channels, self.config.data.image_size)
                real_im = stretch_image(real, self.config.data.channels, self.config.data.image_size)
                futr_im = stretch_image(futr, self.config.data.channels, self.config.data.image_size)
                padding_hor = 0.5*torch.ones(*real_im.shape[:-1], 2)
                real_data = torch.cat([cond_im, padding_hor, real_im, padding_hor, futr_im], dim=-1)
                pred_data = torch.cat([0.5*torch.ones_like(cond_im), padding_hor, pred_im, padding_hor, 0.5*torch.ones_like(futr_im)], dim=-1)
                padding_ver = 0.5*torch.ones(*real_im.shape[:-2], 2, real_data.shape[-1])
                data = torch.cat([real_data, padding_ver, pred_data], dim=-2)
                # Save
                nrow = ceil(np.sqrt((self.config.data.num_frames_cond+self.config.sampling.num_frames_pred+future)*pred.shape[0])/(self.config.data.num_frames_cond+self.config.sampling.num_frames_pred+future))
                image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
                if train:
                    save_image(image_grid, os.path.join(self.args.log_sample_path, f"videos_stretch_interp_{ckpt}_{i}.png"))
                else:
                    save_image(image_grid, os.path.join(self.args.video_folder, f"videos_stretch_interp_{ckpt}_{i}.png"))

            def save_gen(pred):
                if pred is None:
                    return
                if train:
                    torch.save({"gen": pred}, os.path.join(self.args.log_sample_path, f"videos_gen_{ckpt}.pt"))
                else:
                    torch.save({"gen": pred}, os.path.join(self.args.video_folder, f"videos_gen_{ckpt}.pt"))
                data = stretch_image(pred, self.config.data.channels, self.config.data.image_size)
                # Save
                nrow = ceil(np.sqrt((self.config.data.num_frames_cond+self.config.sampling.num_frames_pred)*pred.shape[0])/(self.config.data.num_frames_cond+self.config.sampling.num_frames_pred))
                image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
                if train:
                    save_image(image_grid, os.path.join(self.args.log_sample_path, f"videos_stretch_gen_{ckpt}_{i}.png"))
                else:
                    save_image(image_grid, os.path.join(self.args.video_folder, f"videos_stretch_gen_{ckpt}_{i}.png"))

            if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
                save_pred(pred, real)

            elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation
                save_interp(pred, real)

            elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred
                save_interp(pred, real)
                save_pred(pred2, real2)

            elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen
                save_pred(pred, real)
                save_gen(pred_uncond)

            elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:     # (1) Interp + (2) Pred + (3) Gen
                save_interp(pred, real)
                save_pred(pred2, real2)
                save_gen(pred_uncond)

            elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:     # (1) Interp + (2) Pred + (3) Gen
                save_interp(pred, real)
                save_gen(pred_uncond)

    if no_metrics:
        return None

    # Calc MSE, PSNR, SSIM, LPIPS
    mse_list = np.array(vid_mse).reshape(-1, preds_per_test).min(-1)
    psnr_list = (10 * np.log10(1 / np.array(vid_mse))).reshape(-1, preds_per_test).max(-1)
    ssim_list = np.array(vid_ssim).reshape(-1, preds_per_test).max(-1)
    lpips_list = np.array(vid_lpips).reshape(-1, preds_per_test).min(-1)

    def image_metric_stuff(metric):
        avg_metric, std_metric = metric.mean().item(), metric.std().item()
        conf95_metric = avg_metric - float(st.norm.interval(0.95, loc=avg_metric, scale=st.sem(metric))[0])
        return avg_metric, std_metric, conf95_metric

    avg_mse, std_mse, conf95_mse = image_metric_stuff(mse_list)
    avg_psnr, std_psnr, conf95_psnr = image_metric_stuff(psnr_list)
    avg_ssim, std_ssim, conf95_ssim = image_metric_stuff(ssim_list)
    avg_lpips, std_lpips, conf95_lpips = image_metric_stuff(lpips_list)

    vid_metrics = {'ckpt': ckpt, 'preds_per_test': preds_per_test,
                    'mse': avg_mse, 'mse_std': std_mse, 'mse_conf95': conf95_mse,
                    'psnr': avg_psnr, 'psnr_std': std_psnr, 'psnr_conf95': conf95_psnr,
                    'ssim': avg_ssim, 'ssim_std': std_ssim, 'ssim_conf95': conf95_ssim,
                    'lpips': avg_lpips, 'lpips_std': std_lpips, 'lpips_conf95': conf95_lpips}

    def fvd_stuff(fake_embeddings, real_embeddings):
        avg_fvd = frechet_distance(fake_embeddings, real_embeddings)
        if preds_per_test > 1:
            fvds_list = []
            # Calc FVD for 5 random trajs (each), and average that FVD
            trajs = np.random.choice(np.arange(preds_per_test), (preds_per_test,), replace=False)
            for traj in trajs:
                fvds_list.append(frechet_distance(fake_embeddings[traj::preds_per_test], real_embeddings))
            fvd_traj_mean, fvd_traj_std  = float(np.mean(fvds_list)), float(np.std(fvds_list))
            fvd_traj_conf95 = fvd_traj_mean - float(st.norm.interval(0.95, loc=fvd_traj_mean, scale=st.sem(fvds_list))[0])
        else:
            fvd_traj_mean, fvd_traj_std, fvd_traj_conf95 = -1, -1, -1
        return avg_fvd, fvd_traj_mean, fvd_traj_std, fvd_traj_conf95

    # Calc FVD
    if calc_fvd1 or calc_fvd2 or calc_fvd3:

        if calc_fvd1:
            # (1) Video Pred/Interp
            real_embeddings = np.concatenate(real_embeddings)
            fake_embeddings = np.concatenate(fake_embeddings)
            avg_fvd, fvd_traj_mean, fvd_traj_std, fvd_traj_conf95 = fvd_stuff(fake_embeddings, real_embeddings)
            vid_metrics.update({'fvd': avg_fvd, 'fvd_traj_mean': fvd_traj_mean, 'fvd_traj_std': fvd_traj_std, 'fvd_traj_conf95': fvd_traj_conf95})

    if second_calc:
        mse2 = np.array(vid_mse2).reshape(-1, preds_per_test).min(-1)
        psnr2 = (10 * np.log10(1 / np.array(vid_mse2))).reshape(-1, preds_per_test).max(-1)
        ssim2 = np.array(vid_ssim2).reshape(-1, preds_per_test).max(-1)
        lpips2 = np.array(vid_lpips2).reshape(-1, preds_per_test).min(-1)

        avg_mse2, std_mse2, conf95_mse2 = image_metric_stuff(mse2)
        avg_psnr2, std_psnr2, conf95_psnr2 = image_metric_stuff(psnr2)
        avg_ssim2, std_ssim2, conf95_ssim2 = image_metric_stuff(ssim2)
        avg_lpips2, std_lpips2, conf95_lpips2 = image_metric_stuff(lpips2)

        vid_metrics.update({'mse2': avg_mse2, 'mse2_std': std_mse2, 'mse2_conf95': conf95_mse2,
                            'psnr2': avg_psnr2, 'psnr2_std': std_psnr2, 'psnr2_conf95': conf95_psnr2,
                            'ssim2': avg_ssim2, 'ssim2_std': std_ssim2, 'ssim2_conf95': conf95_ssim2,
                            'lpips2': avg_lpips2, 'lpips2_std': std_lpips2, 'lpips2_conf95': conf95_lpips2})

        # (2) Video Pred if 1 was Interp
        if calc_fvd2:
            real_embeddings2 = np.concatenate(real_embeddings2)
            fake_embeddings2 = np.concatenate(fake_embeddings2)
            avg_fvd2, fvd2_traj_mean, fvd2_traj_std, fvd2_traj_conf95 = fvd_stuff(fake_embeddings2, real_embeddings2)
            vid_metrics.update({'fvd2': avg_fvd2, 'fvd2_traj_mean': fvd2_traj_mean, 'fvd2_traj_std': fvd2_traj_std, 'fvd2_traj_conf95': fvd2_traj_conf95})

    # (3) uncond
    if calc_fvd3:
        real_embeddings_uncond = np.concatenate(real_embeddings_uncond)
        fake_embeddings_uncond = np.concatenate(fake_embeddings_uncond)
        avg_fvd3, fvd3_traj_mean, fvd3_traj_std, fvd3_traj_conf95 = fvd_stuff(fake_embeddings_uncond, real_embeddings_uncond)
        vid_metrics.update({'fvd3': avg_fvd3, 'fvd3_traj_mean': fvd3_traj_mean, 'fvd3_traj_std': fvd3_traj_std, 'fvd3_traj_conf95': fvd3_traj_conf95})

    if not train and (calc_fvd1 or calc_fvd2 or calc_fvd3):
        np.savez(os.path.join(self.args.video_folder, f"video_embeddings_{ckpt}.npz"),
                    real_embeddings=real_embeddings,
                    fake_embeddings=fake_embeddings,
                    real_embeddings2=real_embeddings2,
                    fake_embeddings2=fake_embeddings2,
                    real_embeddings3=real_embeddings_uncond,
                    fake_embeddings3=fake_embeddings_uncond)

    if train:
        elapsed = str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3]
    else:
        elapsed = str(datetime.timedelta(seconds=(time.time() - self.start_time)))[:-3]
    format_p = lambda dd : ", ".join([f"{k}:{v:.4f}" if k != 'ckpt' and k != 'preds_per_test' and k != 'time' else f"{k}:{v:7d}" if k == 'ckpt' else f"{k}:{v:3d}" if k == 'preds_per_test' else f"{k}:{v}" for k, v in dd.items()])
    logging.info(f"elapsed: {elapsed}, {format_p(vid_metrics)}")
    logging.info(f"elapsed: {elapsed}, mem:{get_proc_mem():.03f}GB, GPUmem: {get_GPU_mem():.03f}GB")

    if train:
        return vid_metrics

    else:

        logging.info(f"elapsed: {elapsed}, Writing metrics to {os.path.join(self.args.video_folder, 'vid_metrics.yml')}")
        vid_metrics['time'] = elapsed

        if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction

            vid_metrics['pred_mse'], vid_metrics['pred_psnr'], vid_metrics['pred_ssim'], vid_metrics['pred_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
            vid_metrics['pred_mse_std'], vid_metrics['pred_psnr_std'], vid_metrics['pred_ssim_std'], vid_metrics['pred_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
            vid_metrics['pred_mse_conf95'], vid_metrics['pred_psnr_conf95'], vid_metrics['pred_ssim_conf95'], vid_metrics['pred_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
            if calc_fvd1:
                vid_metrics['pred_fvd'], vid_metrics['pred_fvd_traj_mean'], vid_metrics['pred_fvd_traj_std'], vid_metrics['pred_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

        elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation

            vid_metrics['interp_mse'], vid_metrics['interp_psnr'], vid_metrics['interp_ssim'], vid_metrics['interp_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
            vid_metrics['interp_mse_std'], vid_metrics['interp_psnr_std'], vid_metrics['interp_ssim_std'], vid_metrics['interp_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
            vid_metrics['interp_mse_conf95'], vid_metrics['interp_psnr_conf95'], vid_metrics['interp_ssim_conf95'], vid_metrics['interp_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
            if calc_fvd1:
                vid_metrics['interp_fvd'], vid_metrics['interp_fvd_traj_mean'], vid_metrics['interp_fvd_traj_std'], vid_metrics['interp_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

        elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred

            vid_metrics['interp_mse'], vid_metrics['interp_psnr'], vid_metrics['interp_ssim'], vid_metrics['interp_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
            vid_metrics['interp_mse_std'], vid_metrics['interp_psnr_std'], vid_metrics['interp_ssim_std'], vid_metrics['interp_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
            vid_metrics['interp_mse_conf95'], vid_metrics['interp_psnr_conf95'], vid_metrics['interp_ssim_conf95'], vid_metrics['interp_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
            if calc_fvd1:
                vid_metrics['interp_fvd'], vid_metrics['interp_fvd_traj_mean'], vid_metrics['interp_fvd_traj_std'], vid_metrics['interp_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

            if second_calc:
                vid_metrics['pred_mse'], vid_metrics['pred_psnr'], vid_metrics['pred_ssim'], vid_metrics['pred_lpips'] = vid_metrics['mse2'], vid_metrics['psnr2'], vid_metrics['ssim2'], vid_metrics['lpips2']
                vid_metrics['pred_mse_std'], vid_metrics['pred_psnr_std'], vid_metrics['pred_ssim_std'], vid_metrics['pred_lpips_std'] = vid_metrics['mse2_std'], vid_metrics['psnr2_std'], vid_metrics['ssim2_std'], vid_metrics['lpips2_std']
                vid_metrics['pred_mse_conf95'], vid_metrics['pred_psnr_conf95'], vid_metrics['pred_ssim_conf95'], vid_metrics['pred_lpips_conf95'] = vid_metrics['mse2_conf95'], vid_metrics['psnr2_conf95'], vid_metrics['ssim2_conf95'], vid_metrics['lpips2_conf95']
            if calc_fvd2:
                    vid_metrics['pred_fvd'], vid_metrics['pred_fvd_traj_mean'], vid_metrics['pred_fvd_traj_std'], vid_metrics['pred_fvd_traj_conf95'] = vid_metrics['fvd2'], vid_metrics['fvd2_traj_mean'], vid_metrics['fvd2_traj_std'], vid_metrics['fvd2_traj_conf95']

        elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen

            vid_metrics['pred_mse'], vid_metrics['pred_psnr'], vid_metrics['pred_ssim'], vid_metrics['pred_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
            vid_metrics['pred_mse_std'], vid_metrics['pred_psnr_std'], vid_metrics['pred_ssim_std'], vid_metrics['pred_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
            vid_metrics['pred_mse_conf95'], vid_metrics['pred_psnr_conf95'], vid_metrics['pred_ssim_conf95'], vid_metrics['pred_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
            if calc_fvd1:
                vid_metrics['pred_fvd'], vid_metrics['pred_fvd_traj_mean'], vid_metrics['pred_fvd_traj_std'], vid_metrics['pred_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

            if calc_fvd3:
                vid_metrics['gen_fvd'], vid_metrics['gen_fvd_traj_mean'], vid_metrics['gen_fvd_traj_std'], vid_metrics['gen_fvd_traj_conf95'] = vid_metrics['fvd3'], vid_metrics['fvd3_traj_mean'], vid_metrics['fvd3_traj_std'], vid_metrics['fvd3_traj_conf95']

        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:     # (1) Interp + (2) Pred + (3) Gen

            vid_metrics['interp_mse'], vid_metrics['interp_psnr'], vid_metrics['interp_ssim'], vid_metrics['interp_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
            vid_metrics['interp_mse_std'], vid_metrics['interp_psnr_std'], vid_metrics['interp_ssim_std'], vid_metrics['interp_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
            vid_metrics['interp_mse_conf95'], vid_metrics['interp_psnr_conf95'], vid_metrics['interp_ssim_conf95'], vid_metrics['interp_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
            if calc_fvd1:
                vid_metrics['interp_fvd'], vid_metrics['interp_fvd_traj_mean'], vid_metrics['interp_fvd_traj_std'], vid_metrics['interp_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

            if second_calc:
                vid_metrics['pred_mse'], vid_metrics['pred_psnr'], vid_metrics['pred_ssim'], vid_metrics['pred_lpips'] = vid_metrics['mse2'], vid_metrics['psnr2'], vid_metrics['ssim2'], vid_metrics['lpips2']
                vid_metrics['pred_mse_std'], vid_metrics['pred_psnr_std'], vid_metrics['pred_ssim_std'], vid_metrics['pred_lpips_std'] = vid_metrics['mse2_std'], vid_metrics['psnr2_std'], vid_metrics['ssim2_std'], vid_metrics['lpips2_std']
                vid_metrics['pred_mse_conf95'], vid_metrics['pred_psnr_conf95'], vid_metrics['pred_ssim_conf95'], vid_metrics['pred_lpips_conf95'] = vid_metrics['mse2_conf95'], vid_metrics['psnr2_conf95'], vid_metrics['ssim2_conf95'], vid_metrics['lpips2_conf95']
                if calc_fvd2:
                    vid_metrics['pred_fvd'], vid_metrics['pred_fvd_traj_mean'], vid_metrics['pred_fvd_traj_std'], vid_metrics['pred_fvd_traj_conf95'] = vid_metrics['fvd2'], vid_metrics['fvd2_traj_mean'], vid_metrics['fvd2_traj_std'], vid_metrics['fvd2_traj_conf95']

            if calc_fvd3:
                vid_metrics['gen_fvd'], vid_metrics['gen_fvd_traj_mean'], vid_metrics['gen_fvd_traj_std'], vid_metrics['gen_fvd_traj_conf95'] = vid_metrics['fvd3'], vid_metrics['fvd3_traj_mean'], vid_metrics['fvd3_traj_std'], vid_metrics['fvd3_traj_conf95']

        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:  # (1) Interp + (3) Gen

            vid_metrics['interp_mse'], vid_metrics['interp_psnr'], vid_metrics['interp_ssim'], vid_metrics['interp_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
            vid_metrics['interp_mse_std'], vid_metrics['interp_psnr_std'], vid_metrics['interp_ssim_std'], vid_metrics['interp_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
            vid_metrics['interp_mse_conf95'], vid_metrics['interp_psnr_conf95'], vid_metrics['interp_ssim_conf95'], vid_metrics['interp_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
            if calc_fvd1:
                vid_metrics['interp_fvd'], vid_metrics['interp_fvd_traj_mean'], vid_metrics['interp_fvd_traj_std'], vid_metrics['interp_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

            if calc_fvd3:
                vid_metrics['gen_fvd'], vid_metrics['gen_fvd_traj_mean'], vid_metrics['gen_fvd_traj_std'], vid_metrics['gen_fvd_traj_conf95'] = vid_metrics['fvd3'], vid_metrics['fvd3_traj_mean'], vid_metrics['fvd3_traj_std'], vid_metrics['fvd3_traj_conf95']

        logging.info(f"elapsed: {elapsed}, {format_p(vid_metrics)}")
        write_to_yaml(os.path.join(self.args.video_folder, 'vid_metrics.yml'), vid_metrics)
