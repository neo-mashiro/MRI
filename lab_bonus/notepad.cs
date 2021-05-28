/*
1. 先写一个Otsu方法把阈值T算出来，记下来 (参考wiki就足够了，看一下以前python是怎么算的直方图，代码抄过来，
忽略黑像素即Data中的0，对非0像素做二元分类，只有两个class，忽略的意思是，在for循环中continue跳过，但不要改动Data的数据)

2. 然后去研究marching cubes算法，理解透彻，写一个Unity脚本(继承Monobehavior，脚本名叫MarchingCubes.cs)
实现marching cubes算法 (作为一个static静态函数) (继承Mono是因为算法会需要用到Unity的函数来计算mesh，包括vertices和normal等)

3. 思考怎样把brain的Data + 配合上阈值T + Coordinates，转换成marching cubes算法的input变量，并不改动到Data的原值

4. 写一个脚本MainMenu.cs，设置SerializedField在Start中关联到UI的按钮上，按钮为LoadImage，GenerateMesh，Reset。每个按钮会
触发对应的函数调用，比如，LoadImage要去new NiftiImage()得到brain的Data和Coordinates，
GenerateMesh要去调用MarchingCube算法生成mesh，并在scene中instantiate合适的gameobject，给它指定MeshRenderer和默认的material。
MainMenu.cs是挂在一个empty的gameobject上的，相当于游戏的主入口，其实可以挂在UI的Canvas上。

5. 大脑的mesh如果显示正常，基本上可以了，然后写一个简单的MouseController.cs的脚本，用鼠标拖拽来rotate大脑。
然后再写一个简单的CameraController.cs的脚本，camera位置不变始终在房间内部上帝视角，只实现鼠标zoom的功能即可

6. build a simple room (Cornell box), add point light, bake light (参考CS512的光照烘培那一章)
7. UI上加一个按钮，用于变换mesh的颜色，让玩家可以自由选择，高级点可以写几个shader
8. UI上再加一个slider，可以用鼠标拖拽调节mesh的alpha透明度，看到大脑的内部。
9. 最后一步了，加上fps counter，参考mana oasis，完工，开始写README文档


// study marching cubes, reference code on Github (already starred)
videos yet to watch:
https://www.youtube.com/watch?v=JdeyNbDACV0
https://www.youtube.com/watch?v=B_xk71YopsA
https://www.youtube.com/watch?v=5g7sL1RUu1I
https://www.youtube.com/watch?v=fstxQiZeKJ8
https://www.youtube.com/watch?v=GWmvHPbG0zY
https://zhuanlan.zhihu.com/p/62412131
https://en.wikipedia.org/wiki/Marching_cubes
*/


/* README.md

thresholding is sensitive to noise

to successfully apply marching cubes on the array
our brain data must be skull-stripped and denoised first (we assume they are already done)
otherwise, the grey matter will be shadowed by the skull so the mesh is invisible
just make sure that the input image to marching cubes are completely clean and preprocessed

Pixel intensities of a structure of interest often vary little throughout the structure
Hence, can easily apply a threshold to pixel values to segment the structure of interest (grey matter in our case)




















































*/
