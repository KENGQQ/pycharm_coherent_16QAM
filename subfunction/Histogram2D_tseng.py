import plotly.graph_objects as go
import numpy as np

def Histogram2D_tseng(filename, data, SNR=0, EVM=0, bercount=(0, 0, 0)):
    x = np.real(data)
    y = np.imag(data)
    miny = y.min()
    fig = go.Figure()
    filename = str(filename)
    # fig.add_trace(go.Histogram2dContour(
    #     x=x,
    #     y=y,
    #     colorscale='Hot',
    #     reversescale=True,
    #     xaxis='x',
    #     yaxis='y'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y,
    #     xaxis='x',
    #     yaxis='y',
    #     mode='markers',
    #     marker=dict(
    #         color='rgba(255,156,0,1)',
    #         size=3)
    # ))
    # fig.add_trace(go.Histogram(
    #     y=y,
    #     xaxis='x2',
    #     marker=dict(
    #         color='#F58518'
    #     )
    # ))
    # fig.add_trace(go.Histogram(
    #     x=x,
    #     yaxis='y2',
    #     marker=dict(
    #         color='#F58518'
    #     )
    # ))
    # if SNR != 0:
    #     fig.add_annotation(
    #         text='SNR:{:.2f}(dB)<br>EVM:{:.2f}(%)<br>Bercount:{:.2E} [{}/{}]'.format(SNR, EVM * 100, bercount[0],
    #                                                                                        bercount[1], bercount[2]),
    #         align='left',
    #         showarrow=False,
    #         font_family='Arial',
    #         font_size=17,
    #         font_color='white',
    #         bgcolor='black',
    #         # xref='x2',
    #         x=0,
    #         y=miny - 0.3,
    #         bordercolor='orange',
    #         borderwidth=5
    #         )
        # fig.add_annotation(
        #     x=0,
        #     y=miny-0.3,
        #     text="SNR = {:.2f}(dB)".format(SNR),
        #     showarrow=False)
    fig.add_trace(go.Histogram2d(
        x=x,
        y=y,
        colorscale=[[0.0, "rgb(255, 255, 255)"],
                [0.1111111111111111, "rgb(255,230,0)"],
                [0.2222222222222222, "rgb(255,150,0)"],
                [0.3333333333333333, "rgb(255,102,0)"],
                [0.4444444444444444, "rgb(254,051,0)"],
                [0.5555555555555556, "rgb(255,0,0)"],
                [0.6666666666666666, "rgb(255,0,51)"],
                [0.7777777777777778, "rgb(153,0,51)"],
                [0.8888888888888888, "rgb(60,0,30)"],
                [1.0, "rgb(35,0, 0)"]],
        nbinsx=256,
        nbinsy=256,
        zauto=True,
    ))
    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 1],
            fixedrange=True,
            mirror=True,
            ticks='inside',
            showline=True,
            linewidth=2, linecolor='black',
            showticklabels=False
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 1],
            fixedrange=True,
            mirror=True,
            ticks='inside',
            showline=True,
            linewidth=2, linecolor='black',
            showticklabels=False
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.905, 1],
            fixedrange=True,
            mirror=True,
            ticks='inside',
            showline=True,
            linewidth=2, linecolor='black',
            showticklabels=False
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.905, 1],
            fixedrange=True,
            mirror=True,
            ticks='inside',
            showline=True,
            linewidth=2, linecolor='black',
            showticklabels=False
        ),
        height=800,
        width=800,
        bargap=0,
        hovermode='closest',
        showlegend=False,
        title=go.layout.Title(text="Color Histogram---" + filename),
        font=dict(
            family="Arial",
            size=20,
            color="Black"),
        # yaxis_range=[-4.3, 4.3],
        # xaxis_range=[-4.3, 4.3]
        )
    if SNR != 0:
        fig.update_layout(
            xaxis=dict(
                zeroline=False,
                domain=[0, 0.9],
                showgrid=True,
                fixedrange=True,
                title='In-Phase<br>SNR:{:.2f}(dB) || EVM:{:.2f}(%) || Bercount:{:.2E} [{}/{}]'.format(SNR, EVM * 100, bercount[0],
                                                                                           bercount[1], bercount[2]),
            ),
            font=dict(
                family="Arial",
                size=20,
                color="Black"),
        )
    fig.write_image("G:\KENG\pycharm\pycharm_coherent_16QAM\data\KENG_optsim_py\{}.pdf".format(filename))
    # fig.write_image("C:/Users/yenhsiang/PycharmProjects/DSP/repo/image/{}.pdf".format(filename))