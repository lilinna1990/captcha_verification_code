# captcha_verification_code
## 国家企业信用信息公示系统破解代码和结果例子
### 企业信用信息官网中含有短语有序顺序的验证码破解
#### 描述：

http://www.gsxt.gov.cn/index.html
在国家企业信用信息公示系统上进行数据爬取时，会通过语序顺序的彩色汉字验证码识别，
该识别需要按照常用短语顺序点击验证图像上的汉字字符，
汉字字符存在，字体不同，旋转，模糊以及在彩色图像的背景中
如下图


图片: https://uploader.shimo.im/f/ISJI0lFFQhUy2rB6.png
图片: https://uploader.shimo.im/f/0DK7VXqlKC4dsgmA.png



#### 关键代码和主要流程说明：
#### 主要流程：
- 整体分为了左下角白色背景区域的识别，与彩色部分识别的结果共同联系获取
图片: https://uploader.shimo.im/f/dxDSvVEDfAw7JRF8.png

- 首先识别到参考文字后，利用参考文字的顺序获取改组词汇的顺序
然后利用彩色文字的识别对应后获得彩色文字的顺序

图片: https://uploader.shimo.im/f/4pIH64RAcQof2Ofu.png
##### 左下角白色部分：
分为：
###### 基于ＹＯＬＯ的单个字体检测：

训练的网络用于获取单个字体的位置，从而分割出来
图片: https://uploader.shimo.im/f/5Etfk1Q1SBI8JYDR.png
获得单个字之后对其检测部分单独做分类
图片: https://uploader.shimo.im/f/wkqavqMqOzsNpC34.png
###### 使用基于ｃｒｎｎ的模型的字符分类：
crnn_chinese_characters_rec-master/gen_printed_char_white_wirtelabel.py
该组代码可以实现模拟左下角的字体生成训练的数据
生成过程中字体有旋转和遮挡等变化，以适应文字的变化。

###### 结巴词汇
参考使用：
https://github.com/fxsjy/jieba

###### 获得语序
图片: https://uploader.shimo.im/f/C083KsOlZtUC6Z41.png


　　　　
##### 彩色部分：
###### 基于ＹＯＬＯ的彩色字体检测：
yolov3-master

训练的网络用于获取单个字体的位置，从而分割出来
图片: https://uploader.shimo.im/f/ajV9dwPmI7MOiLkv.png
获得单个字之后对其检测部分单独做分类

###### 使用基于ｃｒｎｎ的模型的字符分类斜体字体：
/crnn_chinese_characters_rec-master/gen_printed_char_white_wirtelabel-64X64-xieti.py
该组代码可以实现黑白的和ｃｏｌｏｒ中类似的斜体.
图片: https://uploader.shimo.im/f/zTAVeYdX1yk8QfMa.png

###### 彩色字体分割
- 这些字体帮助分类黑白的字体，而彩色验证码需要
通过分割的办法减少背景干扰

图片: https://uploader.shimo.im/f/dltSmlhQyRYBgF0G.png
Ｇ－Ｃｏｐｙ２．ｐｔｈ
是用来分割的网络，是通过wgan获取的
wgan-pytorch

- 后面仍然和左下角的分类类似，不同的是训练集

###### 识别过程：

左下角获取了文字和顺序后，再与彩色部分获取的结果做对照，对应好一一的顺序，后得到最后的结果输出，按照顺序的中间坐标。
图片: https://uploader.shimo.im/f/wVEUvB7Tra85vAeo.png


##### 关键代码说明：
- crnn_model_path 是训练出来的左下角的辅助词汇部分的验证码的单汉字识别的模型的路径
对应模型是model_crnn

- color_crnn_model_path 是训练出来的彩色词汇部分的验证码的单汉字识别的模型的路径
对应

图片: https://uploader.shimo.im/f/M17Q3YNmFDsFH24w.png图片: https://uploader.shimo.im/f/DuCSMB7QyckFj1DC.png
获得左下角的字体内容和顺序

图片: https://uploader.shimo.im/f/H8guSUYgPT09JgUA.png

获得彩色部分的，并结合左下角的结果
图片: https://uploader.shimo.im/f/tjQ3qp4Lpk4YJMyI.png



