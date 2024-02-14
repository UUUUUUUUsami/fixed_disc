from discrimination_fixed import discrimination_fixed
import warnings

warnings.simplefilter('ignore')
print("動画をフォルダ内に置いた後、判別を行う動画名を入力してください(判別可能な動画はシーンの切り替えがない動画に限ります)")
x = input()
df = discrimination_fixed(x)
print("-----------------------------------------")
print("判別結果:",end="")
y=df(True)
print(y)