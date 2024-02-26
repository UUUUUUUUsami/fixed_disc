from Luminance_change_and_motion_change import between_two_image
import warnings
import argparse
warnings.simplefilter('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("img1",help="")
parser.add_argument("img2",help="")
parser.add_argument("--pix",type=int,default=10,help="")
parser.add_argument("--rate",type=int,default=5,help="")
parser.add_argument("--luminance",type=int,default=15,help="")
parser.add_argument("-m","--mode",type=int,choices=[1,2],default=0,help="")

args = parser.parse_args()

bti = between_two_image(args.img1,args.img2,pix=args.pix,rate=args.rate,luminance=args.luminance)
first_bool , second_bool = bti() #first_bool:動きの大きさ,second_bool:輝度の大きさ


print("-----------------------------------------")
print("判別結果:",end="")

if args.mode == 0:
    if first_bool:
        print("2フレーム間に写る物体の動きが大きいです")
    else:
        print("2フレーム間の動きの大きさは特別大きくありません")
    if second_bool:
        print("2フレーム間に写る物体の輝度の変化が特別大きくありません")
    else:
        print("2フレーム間の物体の輝度が大きいです")

elif args.mode == 1:
    if first_bool:
        print("2フレーム間に写る物体の動きが大きいです")
    else:
        print("2フレーム間の動きの大きさは特別大きくありません")

elif args.mode == 2:
    if second_bool:
        print("2フレーム間に写る物体の輝度の変化が特別大きくありません")
    else:
        print("2フレーム間の物体の輝度が大きいです")
