# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:57:14 2017

@author: jimmybow
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event, State
import webbrowser
from flask import Flask
import visdcc
import pandas as pd
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn import tree 
import colorlover as cl
import copy

fea_type = np.array(['num', 'num', 'num', 'num'])
fea_positive =  np.array([True, True, True, True])
iris = datasets.load_iris()
df = pd.DataFrame(np.column_stack((iris.data, iris.target_names[iris.target])), 
                  columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'Species'])

# random_state = 42 確保我們每次執行的結果相同
train_x, test_x, train_y, test_y = train_test_split(df.drop('Species', 1), df.Species , random_state = 42, test_size = 0.3)

# 決策樹 建模
Tree = tree.DecisionTreeClassifier(random_state = 42).fit(train_x, train_y)

# 預測值
pred = Tree.predict(test_x)

# 模型評估
# 混淆矩陣
tb = pd.crosstab(index = test_y, columns = pred, rownames = ['實際值'], colnames = ['預測值'])

# 準確率
acc = np.diag(tb).sum() / tb.as_matrix().sum()

# 支點的顏色
tree_color = np.array(cl.scales['4']['qual']['Accent'])

# 網絡圖設置
ww_split = Tree.tree_.feature >= 0
features = pd.Series([iris.feature_names[i] for i in Tree.tree_.feature])
str_split = features  + ' &le; ' + pd.Series(Tree.tree_.threshold).astype(str)

# 左側的邊 (children_left)  方向是 小於等於 (<=)
# 右側的邊 (children_right) 方向是 大於 (>)
data = {}
data['nodes'] = [{'id' : i, 'hidden': False, 'show_leaf': True,
                  'title' : 'Gini impurity = ' + str(Tree.tree_.impurity[i]) +
                  '<br>samples = ' + str(int(Tree.tree_.weighted_n_node_samples[i])) +
                  '<br>( ' + ', '.join(Tree.classes_ + ' = ' + Tree.tree_.value[i][0].astype(int).astype(str)) + ' )'
                                     }  for i in range(Tree.tree_.node_count)] 

for i in np.where(ww_split)[0]: 
    data['nodes'][i]['title'] = str_split[i] + '<br>' + data['nodes'][i]['title']
    data['nodes'][i]['label'] = '%.3f'%(Tree.tree_.threshold[i])
    data['nodes'][i]['color'] = tree_color[Tree.tree_.feature][i] 
for i in np.where(~ww_split)[0]: 
    data['nodes'][i]['shape'] = 'box'
    ww_max = Tree.tree_.value[i][0].astype(int) == max(Tree.tree_.value[i][0].astype(int))
    data['nodes'][i]['label'] = Tree.classes_[ww_max][0]
    data['nodes'][i]['color'] = np.array(cl.scales['3']['qual']['Set2'])[ww_max][0] 
    
for i in range(Tree.tree_.node_count): 
    data['nodes'][i]['title'] = '<div style = "text-align: center">' + data['nodes'][i]['title'] + '</div>' 
    data['nodes'][i]['fixed'] = {'y': True}    

data['edges'] = [{'id': str(i) + '-' + str(Tree.tree_.children_left[i]),
                  'hidden': False, 
                  'from': i, 
                  'to': Tree.tree_.children_left[i],
                  'color': 'red',
                  'title': 'Yes' } for i in np.where(ww_split)[0] ] + [
                {'id': str(i) + '-' + str(Tree.tree_.children_right[i]),
                  'hidden': False,
                  'from': i, 
                  'to': Tree.tree_.children_right[i],
                  'color': 'blue',
                  'title': 'No' } for i in np.where(ww_split)[0] ]

options = {
  'height': '1200px',
  'layout': {
    'hierarchical': {
      'enabled': True,
      'sortMethod': 'directed'  
    }  
  },
  'interaction': {'hover': True},  
  'nodes': {'shape': 'circle', 'font': {'size': 25}}, 
  'edges':{'arrows': {'to': {'enabled': True}},
           'smooth': {'type': "cubicBezier", 'forceDirection': 'vertical'}},
  'physics':{'barnesHut': {'avoidOverlap': 0.4}}         
}

ww_fea_used = np.isin(iris.feature_names, features)  
n_fea = ww_fea_used.sum()
data_legend = {} 
data_legend['nodes'] = [{'id' : i, 'label': np.array(iris.feature_names)[ww_fea_used][i], 'shape': 'dot', 'size': 120, 
                         'font': {'size': 80}, 'x': 300, 'y': 400*i,
                         'color': tree_color[ww_fea_used][i]}    for i in range(n_fea)]    
data_legend['edges'] = []    
options_legend = {'interaction':{'dragView': False, 'dragNodes': False, 'zoomView': False},
                  'physics': {'enabled': False},
                  'height': 100*n_fea }       

# 錯誤分類表
conf = dict(scrollZoom = True,
            displaylogo= False,
            showLink = False,
            modeBarButtonsToRemove = [
            'sendDataToCloud',
            'zoomIn2d',
            'zoomOut2d',
            'hoverClosestCartesian',
            'hoverCompareCartesian',
            'hoverClosest3d',
            'hoverClosestGeo',
            'resetScale2d'])
  
tb_data = [{'type': 'heatmap',
            'z' : tb.as_matrix(),
            'x' : iris.target_names,
            'y' : iris.target_names,
            'hoverinfo': "z",
            'colorscale' : [[0, 'rgb(255,255,255)'], [1, 'rgb(0, 0, 200)'] ],
            'colorbar' : dict(title = "筆數", titlefont = dict(size = 20))}]              
tb_layout = dict(title = '錯誤分類表：', 
                 xaxis = dict(title = '預測值', titlefont = dict(size = 25)),
                 yaxis = dict(title = '實際值', titlefont = dict(size = 25)), 
                 titlefont = dict(size = 35, color = '#8d5413'),
                 margin = dict(l = 140, b = 60, t = 60),
                 dragmode = "pan"                )
     

#全域
glo = {}
glo['sel'] = 0
glo['b1'] = {'n':0, 'run': False}
glo['b2'] = {'n':0, 'run': False}
        
# Dash 
server = Flask(__name__)
app = dash.Dash(name = __name__, server = server)
app.config.supress_callback_exceptions = True

my_css_url = "https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css"
app.css.append_css({
    "external_url": my_css_url
})

app.layout = html.Div([
      html.Div(id = 'b1'), html.Div(id = 'b2'),    
      html.H1('Iris 鳶尾花品種預測', 
              style = {'color': '#ae6c0d', 'display':'inline-block', 'width': '80%',
                       'text-align': 'center', 'vertical-align':'top'}),
      html.Div([dcc.Dropdown(id = 'choose',
                             options=[{'label': '模型評估', 'value': 'model'},
                                      {'label': '預測', 'value': 'pre'} ],     
                             value = 'model'   )],
                style={'width': '15%', 'margin': '21.44px', 'display':'inline-block', 'font-size': 20}   ), html.Br(),      
      html.Div([  
          visdcc.Network(id = 'net', data = data, options = options,
                         selection = {'nodes':[], 'edges':[]},
                         style = {'width': '82%', 'display':'inline-block', 'vertical-align':'top'}), 
          visdcc.Network(id = 'legend', data = data_legend, options = options_legend,
                         selection = {'nodes':[], 'edges':[]},
                         style = {'width': '18%', 'display':'inline-block', 'vertical-align':'top'})],
      style = {'width': '60%', 'display':'inline-block', 'vertical-align':'top'}),
      html.Div(id = 'right-hand', style = {'width': '40%', 'display':'inline-block', 'text-align': 'center'})       
])    

@app.callback(
    Output('net', 'options'),
    [Input('net', 'id')])
def myfun(x): 
    return({'layout': {'hierarchical': {'enabled': False}  }} )

@app.callback(
    Output('right-hand', 'children'),
    [Input('choose', 'value')])
def myfun(x): 
    if x == 'model': return(
             [html.H1('模型評估：', style = {'color': '#6a04ae'}),
              html.H2('Classification And Regression Tree (CART)', style = {'color': 'blue'}),
              html.H2('Training data： {} 筆 (70 %)'.format(len(train_x)), style = {'color': 'blue'}),
              html.H2('Testing data： {} 筆 (30 %)'.format(len(test_x)), style = {'color': 'blue'}),
              dcc.Graph(id = 'tb', figure = {'data': tb_data, 'layout': tb_layout}, config = conf),
              html.H2('準確率 (Accuracy)： %.2f'%(100*acc) + '%', style = {'color': 'blue'})]    )
    else : 
        fea = np.array(iris.feature_names)[ww_fea_used]
        r = [html.H1('輸入資訊：', style = {'color':'#6a04ae'})]
        for i in range(n_fea):
            r.append(html.B(fea[i], style = {'display':'inline-block', 'margin': 15, 'font-size': 25, 'color':'blue'}))
            if fea_type[ww_fea_used][i] == 'num':
                r.append(dcc.Input(id = 'fea_' + str(i),
                                   placeholder = 'Enter a value...',
                                   type = 'text',
                                   value = '',
                                   style = {'display':'inline-block', 'margin': 15, 'font-size': 18, 'width': '25%'}))
            r.append(html.Br())        
        r.append(html.Button('預測', id = 'button', 
                             style = {'display':'inline-block', 'width': '30%','margin': '20px 10px', 'font-size': 30}))
        r.append(html.Button('原始圖形', id = 'button2', 
                             style = {'display':'inline-block', 'width': '30%','margin': '20px 10px', 'font-size': 30}))
        r.append(html.Br())
        r.append(html.B(id = 'pre_value', style = {'font-size': 25}))  
        
        return(r)          

@app.callback(
    Output('pre_value', 'children'),
    events = [Event('button', 'click')],
    state = [State('fea_' + str(i), 'value') for i in range(n_fea)] )  
def myfun(*x):
    xx = test_x.iloc[0].copy()
    ww = np.where(ww_fea_used)[0]
    for i in range(n_fea):  xx[ww[i]] = x[i]
    try: r = html.Div('預測值： ' + str(Tree.predict(xx)[0]),  style = {'color':'blue'})
    except ValueError: 
        r = html.Div('請輸入數值而非文字',  style = {'color':'red'})    
    return(r)

@app.callback(
    Output('pre_value', 'input'),
    events = [Event('fea_' + str(i), 'change') for i in range(n_fea)],
    state = [State('fea_' + str(i), 'value') for i in range(n_fea)] )  
def myfun(*x): 
    glo['str'] = '-'.join(x)
    return('')

@app.callback(
    Output('right-hand', 'input'),
    [Input('net', 'selection')])
def myfun(sel): 
    if len(sel['nodes']) > 0 :
        iid = sel['nodes'][0]
        state = data['nodes'][iid]['show_leaf'] 
        data['nodes'][iid]['show_leaf'] = not state
        from_id = [data['edges'][i]['from'] for i in range(len(data['edges']))]    
        while True:
            ww = np.where(np.isin(from_id, iid))[0]
            if ww.sum() == 0 : break
            for i in ww:
                data['edges'][i]['hidden'] = state
                to = data['edges'][i]['to']
                data['nodes'][to]['hidden'] = state
                data['nodes'][to]['show_leaf'] = True    
            iid = [data['edges'][i]['to'] for i in ww]    
            
    if glo['sel'] > 100 : glo['sel'] = 0       
    glo['sel'] = glo['sel'] + 1        
    return(glo['sel'])

@app.callback(
    Output('net', 'data'),
    [Input('right-hand', 'input'),
     Input('b1','input'),
     Input('b2','input')])
def myfhgun(sel, bb1, bb2): 
    ddd = copy.deepcopy(data)
    if glo['b1']['run']:
        try:
            x = glo['str'].split('-')
            xx = test_x.iloc[0].copy()
            ww = np.where(ww_fea_used)[0]
            for i in range(n_fea):  xx[ww[i]] = x[i]
            str(Tree.predict(xx)[0])
            path = np.where(Tree.decision_path(xx).toarray())[1]
            path_e = [str(path[i-1]) + '-' + str(path[i])    for i in range(1, len(path))]
            for i in range(Tree.tree_.node_count): 
                if ddd['nodes'][i]['id'] not in path: ddd['nodes'][i]['color'] = 'hsla(0, 0%, 80%, 0.36)'     
            for i in range(len(ddd['edges'])):   
                if ddd['edges'][i]['id'] not in path_e: ddd['edges'][i]['color'] = 'hsla(0, 0%, 80%, 0.36)' 
        except : pass
    return(ddd) 

@app.callback(
    Output('b2', 'input'),
    events = [Event('button2', 'click')])
def myfhgun():        
    glo['b2']['n'] = glo['b2']['n'] + 1   
    glo['b2']['run'] =  True   
    glo['b1']['run'] =  False    
    return(str(glo['b2']))    

@app.callback(
    Output('b1', 'input'),
    events = [Event('button', 'click')])
def myfhgun():        
    glo['b1']['n'] = glo['b2']['n'] + 1   
    glo['b1']['run'] =  True   
    glo['b2']['run'] =  False   
    return(str(glo['b1'])) 

if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000/', new=0, autoraise=True) 
    server.run(debug=True, use_reloader=False)
