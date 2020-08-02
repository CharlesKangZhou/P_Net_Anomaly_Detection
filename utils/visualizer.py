# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       visualizer
   Project Name:    octNet
   Author :         康
   Date:            2018/9/22
-------------------------------------------------
   Change Activity:
                   2018/9/22:
-------------------------------------------------
"""
import visdom
import numpy as np
import torch

import pdb


class Visualizer(object):
    def __init__(self, env='main', port=31430, **kwargs):
        self.viz = visdom.Visdom(env=env, port=port, **kwargs)

        # 画的第几个数，相当于横坐标
        # 比如（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    # plot line
    def plot_multi_win(self, d, loop_flag=None):
        '''
        一次plot多个或者一个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        long_update = True
        if loop_flag == 0:
            long_update = False
        for k, v in d.items():
            self.plot(k, v, long_update)

    def plot_single_win(self, d, win, loop_i=1):
        """
        :param d: dict (name, value) i.e. ('loss', 0.11)
        :param win: only one win
        :param loop_i: i.e. plot testing loss and label
        :return:
        """
        for k, v in d.items():
            index_k = '{}_{}'.format(k, win)
            x = self.index.get(index_k, 0)
            self.viz.line(Y=np.array([v]), X=np.array([x]),
                          name=k,
                          win=win,
                          opts=dict(title=win, showlegend=True),
                          update='append' if (x > 0 and loop_i > 0) else None)
                          # update=None if (x == 0 or loop_i == 0) else 'append')
            self.index[index_k] = x + 1

    def plot_legend(self, win, name, y, long_update=True, **kwargs):
        '''
        plot different line in different time in the same window
        One mame, one win: only one lie in a win.
        '''
        # eg. 
        # self.vis.plot_legend(win='iou', name='val', y=iou.mean())
        x = self.index.get(
            name, 0)  # dict.get(key, default=None). 返回指定键的值，如果值不在字典中返回default值
        self.viz.line(Y=np.array([y]), X=np.array([x]),
                      name=name,
                      win=win,
                      opts=dict(title=win, showlegend=True),
                      update='append' if (x > 0 and long_update) else None,
                      **kwargs)
        self.index[name] = x + 1    # Maintain the X

    def plot(self, name, y, long_update, **kwargs):
        '''
        self.plot('loss', 1.00)
        One mame, one win: only one lie in a win.
        '''
        x = self.index.get(
            name, 0)  # dict.get(key, default=None). 返回指定键的值，如果值不在字典中返回default值
        self.viz.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update='append' if (x > 0 and long_update) else None,
                      **kwargs)
        self.index[name] = x + 1    # Maintain the X

    # TODO_: delete
    def img(self, img_, name, **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        '''
        self.viz.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    # TODO_: delete
    def img_cpu(self, img_, name, **kwargs):
        if not isinstance(img_, np.ndarray):
            img_ = np.array(img_)
        self.viz.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def images(self, images, win_name='train', nrow=8, img_name=None):
        """
        There are images in on win.
        images: single image or concated images
        win_name: the window name
        nrow: number of images in a row
        img_name: window title name

        Example:
        # only show one image in a batch
        images = torch.cat([input[0], gt[0], output[0])
        self.vis.images(images, win_name='train', img_name=img_name[0], nrow=3)
        """
        if img_name is None:
            title = win_name
        else:
            title = '{}_{}'.format(win_name, img_name)
        self.viz.images(
            images,
            nrow=nrow,
            win=win_name,
            opts=dict(title=title, caption=img_name)
        )

    def text(self, text, name='text'):
        self.viz.text(text, win=name)

    def line(self, **kwargs):
        self.viz.line(**kwargs)

    def draw_roc(self, fpr, tpr):
        self.viz.line(Y=np.array(tpr), X=np.array(fpr),
                      name='roc_curve',
                      win='roc_curve',
                      opts=dict(title='roc_curve', showlegend=True))

    def __getattr__(self, name):
        '''
        self.function 等价于self.vis.function
        自定义的plot,image,log,plot_many等除外
        '''
        raise ValueError('wrong in visulizer')
        # return getattr(self.vis, name)


def main():
    vis = Visualizer()
    # vis.line(2, 2, '332')

    # vis = visdom.Visdom(port=31430)
    # vis.line(np.array([2]), np.array([2]))


if __name__ == '__main__':
    main()
