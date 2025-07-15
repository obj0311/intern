import os
import cv2
import argparse
import torch
from torchvision import utils
from face_model.gpen_model import FullGenerator_t


def test(args, model, img_lq, image_name):
    img_lq       = torch.from_numpy(img_lq).to(args.device).permute(2, 0, 1).unsqueeze(0)
    img_lq       = (img_lq/255.-0.5)/0.5
    img_lq       = torch.flip(img_lq, [1])
    with torch.no_grad():
        img_out = model(img_lq)
        utils.save_image(img_out,   os.path.join(args.result_path, '2' + image_name), nrow=1, normalize=True,  value_range=(-1, 1))
        # utils.save_image(img_out,   os.path.join(args.result_path, image_name.replace('jpg', 'png')), nrow=1, normalize=True,  range=(-1, 1))


def main(args):

    if args.gen_script == True:
        model           = FullGenerator_t(args.image_size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=args.device).to(args.device)
        ckpt            = torch.load(args.pretrain, map_location="cuda:0")

        model.load_state_dict(ckpt['g_ema'])
        model.eval()

        traced_script_module = torch.jit.trace(model, torch.rand([1, 3, 512, 512]).to(args.device))
        traced_script_module.save('test.pt')

    else:
        model = torch.jit.load("s2.pt", map_location=args.device)
        image_list     = os.listdir(args.path)
        cnt = 0
        
        for idx, image_name in enumerate(image_list):
            cnt = cnt + 1
            print (cnt, '>>', image_name)
            image       = cv2.imread(args.path+image_name)
            image       = cv2.resize(image,(args.image_size,args.image_size))

            test(args, model, image, image_name)


if __name__ == "__main__":
    modelpath = 'D:/PycharmProjects/StyleTransfer/GPEN_kaist_v1_0901/assets/ckpt/001000.pth'
    parser      = argparse.ArgumentParser()
    parser.add_argument("--path",               type=str,   default='input/')
    parser.add_argument("--result_path",        type=str,   default='result/')
    parser.add_argument("--image_size",         type=int,   default=512)
    parser.add_argument('--channel_multiplier', type=int,   default=2)
    parser.add_argument('--narrow',             type=float, default=1.0)
    parser.add_argument('--latent',             type=int,   default=512)
    parser.add_argument('--n_mlp',              type=int,   default=8)
    parser.add_argument('--pretrain',           type=str,   default='./E_model.pth')
    parser.add_argument('--device',             type=str,   default='cuda:0')
    parser.add_argument('--gen_script',         type=bool,   default=False)
    args    = parser.parse_args()

    os.makedirs(args.result_path, exist_ok=True)

    main(args)
