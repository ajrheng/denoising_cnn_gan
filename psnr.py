def psnr(clean_imgs, denoised_imgs):
  """
  Calculates mean peak signal to noise ratio given batch of clean and corresponding
  noisy images.

  Dimensions of both are [BS x (1) x H x W]
  Channel dimension of 1 is optional as it will be squeezed out
  """
  clean_imgs = clean_imgs.squeeze()
  denoised_imgs = denoised_imgs.squeeze()
    
  # mean over height and width (image)
  mse = torch.mean( (clean_imgs - denoised_imgs) ** 2 , (-2,-1))
  
  # max over height and width (image)
  maxf = torch.max(clean_imgs, -1)
  maxf = torch.max(maxf.values, -1)
  
  # mean psnr over batch
  psnr = torch.mean(20 * torch.log10(1/torch.sqrt(mse)))
  return psnr