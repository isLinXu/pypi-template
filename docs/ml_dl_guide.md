# 深度学习与机器学习扩展指南

本文档提供了在Python库中集成深度学习、机器学习和数据处理功能的详细指南，包括常见框架的使用、模型开发与部署的最佳实践，以及可能遇到的问题和解决方案。

## 目录

- [集成机器学习框架](#集成机器学习框架)
  - [PyTorch集成](#pytorch集成)
  - [TensorFlow集成](#tensorflow集成)
  - [scikit-learn集成](#scikit-learn集成)
- [数据处理与特征工程](#数据处理与特征工程)
  - [大规模数据处理](#大规模数据处理)
  - [特征工程最佳实践](#特征工程最佳实践)
- [模型训练与优化](#模型训练与优化)
  - [分布式训练](#分布式训练)
  - [超参数优化](#超参数优化)
  - [模型压缩与量化](#模型压缩与量化)
- [模型部署与服务](#模型部署与服务)
  - [模型序列化](#模型序列化)
  - [REST API服务](#rest-api服务)
  - [边缘设备部署](#边缘设备部署)
- [常见问题与解决方案](#常见问题与解决方案)
  - [内存管理问题](#内存管理问题)
  - [GPU相关问题](#gpu相关问题)
  - [依赖冲突问题](#依赖冲突问题)

## 集成机器学习框架

### PyTorch集成

在您的Python库中集成PyTorch，首先需要在`pyproject.toml`中添加相关依赖：

```toml
[project.optional-dependencies]
torch = [
    "torch>=1.10.0",
    "torchvision>=0.11.0",
    "torchaudio>=0.10.0",
]
```

然后创建PyTorch模型模块：

```python
# src/your_package_name/models/torch_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class SimpleNN(nn.Module):
    """简单的PyTorch神经网络模型。
    
    这个模型可以用于演示如何在您的包中集成PyTorch模型。
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout: Dropout比率
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数。
        
        Args:
            x: 输入张量，形状为[batch_size, input_dim]
            
        Returns:
            输出张量，形状为[batch_size, output_dim]
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader,
                val_loader: Optional[torch.utils.data.DataLoader] = None,
                epochs: int = 10,
                lr: float = 0.001,
                device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, List[float]]:
    """训练PyTorch模型的通用函数。
    
    Args:
        model: 要训练的PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备（'cuda'或'cpu'）
        
    Returns:
        包含训练历史的字典
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    
    return history
```

### TensorFlow集成

在您的Python库中集成TensorFlow，首先需要在`pyproject.toml`中添加相关依赖：

```toml
[project.optional-dependencies]
tensorflow = [
    "tensorflow>=2.8.0",
    "tensorflow-addons>=0.16.0",
]
```

然后创建TensorFlow模型模块：

```python
# src/your_package_name/models/tf_models.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from typing import Dict, List, Optional, Tuple, Union, Callable

def create_simple_model(input_shape: Tuple[int, ...], 
                        num_classes: int, 
                        hidden_units: List[int] = [128, 64]) -> tf.keras.Model:
    """创建简单的TensorFlow Keras模型。
    
    Args:
        input_shape: 输入形状
        num_classes: 分类数量
        hidden_units: 隐藏层单元数列表
        
    Returns:
        编译好的Keras模型
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
    
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

class CustomModelWrapper:
    """TensorFlow模型的自定义包装器，提供额外功能。
    
    这个包装器可以扩展TensorFlow模型的功能，例如添加早停、学习率调度等。
    
    Args:
        model: TensorFlow Keras模型
        learning_rate: 初始学习率
    """
    def __init__(self, model: tf.keras.Model, learning_rate: float = 0.001):
        self.model = model
        self.learning_rate = learning_rate
        self._setup_optimizer()
    
    def _setup_optimizer(self) -> None:
        """设置优化器。"""
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics
        )
    
    def train(self, 
              train_dataset: tf.data.Dataset, 
              validation_dataset: Optional[tf.data.Dataset] = None, 
              epochs: int = 10, 
              callbacks: List[tf.keras.callbacks.Callback] = None) -> tf.keras.callbacks.History:
        """训练模型。
        
        Args:
            train_dataset: 训练数据集
            validation_dataset: 验证数据集
            epochs: 训练轮数
            callbacks: Keras回调函数列表
            
        Returns:
            训练历史
        """
        default_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        if callbacks:
            default_callbacks.extend(callbacks)
        
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=default_callbacks
        )
        
        return history
    
    def save(self, filepath: str) -> None:
        """保存模型。
        
        Args:
            filepath: 保存路径
        """
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str, custom_objects: Optional[Dict] = None) -> 'CustomModelWrapper':
        """加载模型。
        
        Args:
            filepath: 模型文件路径
            custom_objects: 自定义对象字典
            
        Returns:
            加载的模型包装器
        """
        model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        return cls(model)
```

### scikit-learn集成

在您的Python库中集成scikit-learn，首先需要在`pyproject.toml`中添加相关依赖：

```toml
[project.optional-dependencies]
ml = [
    "scikit-learn>=1.0.0",
    "pandas>=1.3.0",
    "numpy>=1.20.0",
]
```

然后创建scikit-learn模型模块：

```python
# src/your_package_name/models/sklearn_models.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    """自定义特征选择器。
    
    这个转换器可以根据指定的规则选择特征。
    
    Args:
        feature_names: 要选择的特征名称列表
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'CustomFeatureSelector':
        """拟合转换器。
        
        Args:
            X: 输入特征DataFrame
            y: 目标变量（可选）
            
        Returns:
            拟合后的转换器
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据。
        
        Args:
            X: 输入特征DataFrame
            
        Returns:
            转换后的DataFrame
        """
        return X[self.feature_names]

def create_preprocessing_pipeline(numeric_features: List[str], 
                                 categorical_features: List[str]) -> ColumnTransformer:
    """创建预处理管道。
    
    Args:
        numeric_features: 数值特征列表
        categorical_features: 分类特征列表
        
    Returns:
        列转换器
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def create_model_pipeline(preprocessor: ColumnTransformer, 
                         model_type: str = 'classifier', 
                         **model_params) -> Pipeline:
    """创建完整的模型管道。
    
    Args:
        preprocessor: 预处理器
        model_type: 模型类型，'classifier'或'regressor'
        **model_params: 模型参数
        
    Returns:
        完整的管道
    """
    if model_type == 'classifier':
        model = RandomForestClassifier(**model_params)
    else:
        model = GradientBoostingRegressor(**model_params)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def optimize_hyperparameters(pipeline: Pipeline, 
                            param_grid: Dict[str, List[Any]], 
                            X_train: pd.DataFrame, 
                            y_train: np.ndarray,
                            cv: int = 5,
                            scoring: Optional[str] = None) -> GridSearchCV:
    """优化模型超参数。
    
    Args:
        pipeline: 模型管道
        param_grid: 参数网格
        X_train: 训练特征
        y_train: 训练标签
        cv: 交叉验证折数
        scoring: 评分方法
        
    Returns:
        网格搜索结果
    """
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳得分: {grid_search.best_score_:.4f}")
    
    return grid_search

def save_model(model: Any, filepath: str) -> None:
    """保存模型到文件。
    
    Args:
        model: 要保存的模型
        filepath: 保存路径
    """
    joblib.dump(model, filepath)
    print(f"模型已保存到 {filepath}")

def load_model(filepath: str) -> Any:
    """从文件加载模型。
    
    Args:
        filepath: 模型文件路径
        
    Returns:
        加载的模型
    """
    model = joblib.load(filepath)
    print(f"模型已从 {filepath} 加载")
    return model
```

## 数据处理与特征工程

### 大规模数据处理

处理大规模数据集时，内存管理和计算效率是关键。以下是一些处理大规模数据的策略和工具：

#### 使用Dask进行分布式计算

在`pyproject.toml`中添加依赖：

```toml
[project.optional-dependencies]
big_data = [
    "dask>=2022.1.0",
    "distributed>=2022.1.0",
]
```

创建Dask处理模块：

```python
# src/your_package_name/data/dask_processing.py
import dask.dataframe as dd
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
import os

def load_large_dataset(file_pattern: str, 
                      file_type: str = 'csv', 
                      **kwargs) -> dd.DataFrame:
    """加载大型数据集到Dask DataFrame。
    
    Args:
        file_pattern: 文件路径模式，例如'data/*.csv'
        file_type: 文件类型，支持'csv'、'parquet'等
        **kwargs: 传递给读取函数的参数
        
    Returns:
        Dask DataFrame
    """
    if file_type == 'csv':
        return dd.read_csv(file_pattern, **kwargs)
    elif file_type == 'parquet':
        return dd.read_parquet(file_pattern, **kwargs)
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")

def process_large_dataset(ddf: dd.DataFrame, 
                         operations: List[Callable[[dd.DataFrame], dd.DataFrame]]) -> dd.DataFrame:
    """对大型数据集应用一系列操作。
    
    Args:
        ddf: 输入Dask DataFrame
        operations: 要应用的操作列表，每个操作都是接受DataFrame并返回DataFrame的函数
        
    Returns:
        处理后的Dask DataFrame
    """
    result = ddf
    for op in operations:
        result = op(result)
    return result

def save_large_dataset(ddf: dd.DataFrame, 
                      output_path: str, 
                      file_type: str = 'parquet',
                      partition_cols: Optional[List[str]] = None) -> None:
    """保存大型数据集。
    
    Args:
        ddf: 要保存的Dask DataFrame
        output_path: 输出路径
        file_type: 输出文件类型
        partition_cols: 分区列名列表
    """
    if file_type == 'csv':
        ddf.to_csv(output_path, index=False)
    elif file_type == 'parquet':
        ddf.to_parquet(output_path, partition_on=partition_cols)
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")
    
    print(f"数据已保存到 {output_path}")

def create_chunked_processor(chunk_size: int = 10000) -> Callable:
    """创建一个处理大型Pandas DataFrame的分块处理器。
    
    Args:
        chunk_size: 每个块的大小
        
    Returns:
        分块处理函数
    """
    def process_in_chunks(df: pd.DataFrame, process_func: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """分块处理DataFrame。
        
        Args:
            df: 输入DataFrame
            process_func: 处理函数
            
        Returns:
            处理后的DataFrame
        """
        result_chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            processed_chunk = process_func(chunk)
            result_chunks.append(processed_chunk)
        
        return pd.concat(result_chunks, ignore_index=True)
    
    return process_in_chunks
```

#### 使用Ray进行并行计算

在`pyproject.toml`中添加依赖：

```toml
[project.optional-dependencies]
parallel = [
    "ray>=1.13.0",
]
```

创建Ray并行处理模块：

```python
# src/your_package_name/data/ray_processing.py
import ray
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple

def init_ray(num_cpus: Optional[int] = None) -> None:
    """初始化Ray。
    
    Args:
        num_cpus: 使用的CPU核心数，默认使用所有可用核心
    """
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=num_cpus)
    print(f"Ray已初始化，可用资源: {ray.available_resources()}")

@ray.remote
def process_partition(partition_data: pd.DataFrame, 
                     process_func: Callable[[pd.DataFrame], Any]) -> Any:
    """处理数据分区。
    
    Args:
        partition_data: 分区数据
        process_func: 处理函数
        
    Returns:
        处理结果
    """
    return process_func(partition_data)

def parallel_process(df: pd.DataFrame, 
                    process_func: Callable[[pd.DataFrame], Any],
                    num_partitions: int = 10) -> List[Any]:
    """并行处理DataFrame。
    
    Args:
        df: 输入DataFrame
        process_func: 处理函数
        num_partitions: 分区数量
        
    Returns:
        处理结果列表
    """
    if not ray.is_initialized():
        init_ray()
    
    # 将DataFrame分割成多个分区
    partitions = np.array_split(df, num_partitions)
    
    # 并行处理每个分区
    futures = [process_partition.remote(partition, process_func) for partition in partitions]
    
    # 获取结果
    results = ray.get(futures)
    
    return results

def parallel_apply(df: pd.DataFrame, 
                  apply_func: Callable[[pd.Series], Any],
                  axis: int = 0) -> pd.DataFrame:
    """并行应用函数到DataFrame的行或列。
    
    Args:
        df: 输入DataFrame
        apply_func: 应用函数
        axis: 应用轴，0表示行，1表示列
        
    Returns:
        处理后的DataFrame
    """
    if not ray.is_initialized():
        init_ray()
    
    if axis == 0:
        # 按行应用
        @ray.remote
        def apply_to_row(row):
            return apply_func(row)
        
        futures = [apply_to_row.remote(row) for _, row in df.iterrows()]
        results = ray.get(futures)
        
        if isinstance(results[0], pd.Series):
            return pd.DataFrame(results)
        else:
            return pd.Series(results)
    else:
        # 按列应用
        @ray.remote
        def apply_to_column(col_name, col_data):
            return col_name, apply_func(col_data)
        
        futures = [apply_to_column.remote(col_name, df[col_name]) for col_name in df.columns]
        results = dict(ray.get(futures))
        
        return pd.DataFrame(results)
```

### 特征工程最佳实践

特征工程是机器学习中的关键步骤，以下是一些最佳实践和工具：

#### 创建特征工程工具包

```python
# src/your_package_name/features/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """从日期时间列提取特征。
    
    Args:
        datetime_cols: 日期时间列名列表
        extract_features: 要提取的特征列表，可以包含'year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'quarter', 'is_weekend'
        drop_original: 是否删除原始日期时间列
    """
    def __init__(self, datetime_cols: List[str], 
                 extract_features: List[str] = ['year', 'month', 'day', 'dayofweek', 'is_weekend'],
                 drop_original: bool = True):
        self.datetime_cols = datetime_cols
        self.extract_features = extract_features
        self.drop_original = drop_original
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'DateTimeFeatureExtractor':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col in self.datetime_cols:
            if col in X_copy.columns:
                # 确保列是datetime类型
                X_copy[col] = pd.to_datetime(X_copy[col])
                
                # 提取特征
                if 'year' in self.extract_features:
                    X_copy[f"{col}_year"] = X_copy[col].dt.year
                if 'month' in self.extract_features:
                    X_copy[f"{col}_month"] = X_copy[col].dt.month
                if 'day' in self.extract_features:
                    X_copy[f"{col}_day"] = X_copy[col].dt.day
                if 'hour' in self.extract_features:
                    X_copy[f"{col}_hour"] = X_copy[col].dt.hour
                if 'minute' in self.extract_features:
                    X_copy[f"{col}_minute"] = X_copy[col].dt.minute
                if 'second' in self.extract_features:
                    X_copy[f"{col}_second"] = X_copy[col].dt.second
                if 'dayofweek' in self.extract_features:
                    X_copy[f"{col}_dayofweek"] = X_copy[col].dt.dayofweek
                if 'quarter' in self.extract_features:
                    X_copy[f"{col}_quarter"] = X_copy[col].dt.quarter
                if 'is_weekend' in self.extract_features:
                    X_copy[f"{col}_is_weekend"] = (X_copy[col].dt.dayofweek >= 5).astype(int)
                
                # 删除原始列
                if self.drop_original:
                    X_copy = X_copy.drop(col, axis=1)
        
        return X_copy

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """从文本列提取特征。
    
    Args:
        text_cols: 文本列名列表
        extract_features: 要提取的特征列表，可以包含'length', 'word_count', 'unique_word_count'
        drop_original: 是否删除原始文本列
    """
    def __init__(self, text_cols: List[str], 
                 extract_features: List[str] = ['length', 'word_count'],
                 drop_original: bool = True):
        self.text_cols = text_cols
        self.extract_features = extract_features
        self.drop_original = drop_original
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'TextFeatureExtractor':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col in self.text_cols:
            if col in X_copy.columns:
                # 确保列是字符串类型
                X_copy[col] = X_copy[col].astype(str)
                
                # 提取特征
                if 'length' in self.extract_features:
                    X_copy[f"{col}_length"] = X_copy[col].str.len()
                if 'word_count' in self.extract_features:
                    X_copy[f"{col}_word_count"] = X_copy[col].str.split().str.len()
                if 'unique_word_count' in self.extract_features:
                    X_copy[f"{col}_unique_word_count"] = X_copy[col].apply(lambda x: len(set(x.split())))
                
                # 删除原始列
                if self.drop_original:
                    X_copy = X_copy.drop(col, axis=1)
        
        return X_copy

class FeatureSelector(BaseEstimator, TransformerMixin):
    """特征选择器，支持多种特征选择方法。
    
    Args:
        method: 特征选择方法，支持'k_best', 'pca', 'correlation'
        k: 要选择的特征数量
        score_func: 评分函数，用于k_best方法
        threshold: 相关性阈值，用于correlation方法
    """
    def __init__(self, method: str = 'k_best', 
                 k: int = 10, 
                 score_func: Callable = f_classif,
                 threshold: float = 0.8):
        self.method = method
        self.k = k
        self.score_func = score_func
        self.threshold = threshold
        self.selector = None
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'FeatureSelector':
        if self.method == 'k_best':
            self.selector = SelectKBest(self.score_func, k=self.k)
            self.selector.fit(X, y)
            self.selected_features = X.columns[self.selector.get_support()]
        
        elif self.method == 'pca':
            self.selector = PCA(n_components=self.k)
            self.selector.fit(X)
            self.selected_features = [f'PC{i+1}' for i in range(self.k)]
        
        elif self.method == 'correlation':
            if y is not None:
                # 计算每个特征与目标变量的相关性
                corr = pd.DataFrame()
                for col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        corr.loc[col, 'correlation'] = abs(np.corrcoef(X[col], y)[0, 1])
                
                # 选择相关性高于阈值的特征
                self.selected_features = corr[corr['correlation'] > self.threshold].index.tolist()
                if len(self.selected_features) > self.k:
                    self.selected_features = corr.nlargest(self.k, 'correlation').index.tolist()
            else:
                raise ValueError("correlation方法需要提供目标变量y")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == 'k_best':
            return pd.DataFrame(self.selector.transform(X), columns=self.selected_features)
        
        elif self.method == 'pca':
            return pd.DataFrame(self.selector.transform(X), columns=self.selected_features)
        
        elif self.method == 'correlation':
            return X[self.selected_features]
        
        return X
```

#### 特征工程最佳实践示例

```python
# 特征工程最佳实践示例
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from your_package_name.features.feature_engineering import DateTimeFeatureExtractor, TextFeatureExtractor

# 定义列类型
date_cols = ['order_date', 'delivery_date']
text_cols = ['product_description', 'customer_feedback']
numeric_cols = ['price', 'quantity', 'discount']
categorical_cols = ['category', 'payment_method', 'shipping_method']

# 创建特征工程管道
feature_engineering_pipeline = ColumnTransformer([
    ('date_features', DateTimeFeatureExtractor(date_cols), date_cols),
    ('text_features', TextFeatureExtractor(text_cols), text_cols),
    ('numeric_features', StandardScaler(), numeric_cols),
    ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# 在模型训练管道中使用
from sklearn.ensemble import RandomForestClassifier

model_pipeline = Pipeline([
    ('feature_engineering', feature_engineering_pipeline),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# 训练模型
model_pipeline.fit(X_train, y_train)

# 预测
predictions = model_pipeline.predict(X_test)
```

## 模型训练与优化

### 分布式训练

#### PyTorch分布式训练

在`pyproject.toml`中添加依赖：

```toml
[project.optional-dependencies]
distributed_training = [
    "torch>=1.10.0",
    "torchvision>=0.11.0",
    "horovod>=0.24.0",
]
```

创建PyTorch分布式训练模块：

```python
# src/your_package_name/training/distributed_torch.py
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import os

def setup(rank: int, world_size: int) -> None:
    """设置分布式训练环境。
    
    Args:
        rank: 当前进程的排名
        world_size: 总进程数
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup() -> None:
    """清理分布式训练环境。"""
    dist.destroy_process_group()

def train_model_distributed(model: nn.Module, 
                           train_dataset: torch.utils.data.Dataset,
                           val_dataset: Optional[torch.utils.data.Dataset] = None,
                           epochs: int = 10,
                           batch_size: int = 32,
                           lr: float = 0.001,
                           world_size: int = torch.cuda.device_count()) -> nn.Module:
    """使用分布式训练训练模型。
    
    Args:
        model: 要训练的模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        world_size: 总进程数
        
    Returns:
        训练好的模型
    """
    mp.spawn(
        _train_worker,
        args=(model, train_dataset, val_dataset, epochs, batch_size, lr, world_size),
        nprocs=world_size,
        join=True
    )
    
    # 返回在主进程上训练好的模型
    return model

def _train_worker(rank: int, 
                 model: nn.Module, 
                 train_dataset: torch.utils.data.Dataset,
                 val_dataset: Optional[torch.utils.data.Dataset],
                 epochs: int,
                 batch_size: int,
                 lr: float,
                 world_size: int) -> None:
    """分布式训练工作进程。
    
    Args:
        rank: 当前进程的排名
        model: 要训练的模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        world_size: 总进程数
    """
    setup(rank, world_size)
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=4
        )
    
    # 将模型移动到当前设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 包装模型为DDP模型
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        if val_loader is not None:
            ddp_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = ddp_model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            
            if rank == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            if rank == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    
    # 清理
    cleanup()
```

#### TensorFlow分布式训练

在`pyproject.toml`中添加依赖：

```toml
[project.optional-dependencies]
tf_distributed = [
    "tensorflow>=2.8.0",
    "tensorflow-addons>=0.16.0",
]
```

创建TensorFlow分布式训练模块：

```python
# src/your_package_name/training/distributed_tf.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import os

def setup_mirrored_strategy() -> tf.distribute.MirroredStrategy:
    """设置MirroredStrategy用于多GPU训练。
    
    Returns:
        MirroredStrategy实例
    """
    return tf.distribute.MirroredStrategy()

def setup_tpu_strategy() -> tf.distribute.TPUStrategy:
    """设置TPUStrategy用于TPU训练。
    
    Returns:
        TPUStrategy实例
    """
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)

def train_model_distributed(model_fn: Callable[[], tf.keras.Model],
                           train_dataset: tf.data.Dataset,
                           val_dataset: Optional[tf.data.Dataset] = None,
                           epochs: int = 10,
                           batch_size: int = 32,
                           strategy_type: str = 'mirrored',
                           use_mixed_precision: bool = True) -> tf.keras.Model:
    """使用分布式策略训练模型。
    
    Args:
        model_fn: 返回模型的函数
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        epochs: 训练轮数
        batch_size: 批次大小
        strategy_type: 策略类型，'mirrored'或'tpu'
        use_mixed_precision: 是否使用混合精度训练
        
    Returns:
        训练好的模型
    """
    # 设置混合精度
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("混合精度训练已启用")
    
    # 选择分布式策略
    if strategy_type == 'mirrored':
        strategy = setup_mirrored_strategy()
        print(f"MirroredStrategy已设置，设备: {strategy.num_replicas_in_sync}")
    elif strategy_type == 'tpu':
        strategy = setup_tpu_strategy()
        print(f"TPUStrategy已设置，设备: {strategy.num_replicas_in_sync}")
    else:
        raise ValueError(f"不支持的策略类型: {strategy_type}")
    
    # 准备数据集
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    train_dataset = train_dataset.batch(global_batch_size)
    if val_dataset is not None:
        val_dataset = val_dataset.batch(global_batch_size)
    
    # 在策略范围内创建和编译模型
    with strategy.scope():
        model = model_fn()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # 训练模型
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model
```

### 超参数优化

在`pyproject.toml`中添加依赖：

```toml
[project.optional-dependencies]
hyperparameter_tuning = [
    "optuna>=2.10.0",
    "ray[tune]>=1.13.0",
]
```

创建超参数优化模块：

```python
# src/your_package_name/training/hyperparameter_tuning.py
import optuna
from optuna.integration import TFKerasPruningCallback, PyTorchLightningPruningCallback
import numpy as np
import torch
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def optimize_pytorch_hyperparams(model_class: type,
                               train_fn: Callable,
                               train_dataset: torch.utils.data.Dataset,
                               val_dataset: torch.utils.data.Dataset,
                               param_space: Dict[str, Any],
                               n_trials: int = 20,
                               direction: str = 'maximize',
                               metric: str = 'val_acc') -> Dict[str, Any]:
    """使用Optuna优化PyTorch模型超参数。
    
    Args:
        model_class: 模型类
        train_fn: 训练函数
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        param_space: 参数空间
        n_trials: 试验次数
        direction: 优化方向，'maximize'或'minimize'
        metric: 优化指标
        
    Returns:
        最佳参数
    """
    def objective(trial):
        # 从参数空间采样参数
        params = {}
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
        
        # 创建模型
        model = model_class(**params)
        
        # 训练模型
        result = train_fn(model, train_dataset, val_dataset)
        
        # 返回优化指标
        return result[metric]
    
    # 创建Optuna研究
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    print(f"最佳参数: {study.best_params}")
    print(f"最佳{metric}: {study.best_value}")
    
    return study.best_params

def optimize_tf_hyperparams(model_fn: Callable,
                          train_dataset: tf.data.Dataset,
                          val_dataset: tf.data.Dataset,
                          param_space: Dict[str, Any],
                          n_trials: int = 20,
                          direction: str = 'maximize',
                          metric: str = 'val_accuracy') -> Dict[str, Any]:
    """使用Optuna优化TensorFlow模型超参数。
    
    Args:
        model_fn: 返回模型的函数
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        param_space: 参数空间
        n_trials: 试验次数
        direction: 优化方向，'maximize'或'minimize'
        metric: 优化指标
        
    Returns:
        最佳参数
    """
    def objective(trial):
        # 从参数空间采样参数
        params = {}
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
        
        # 创建模型
        model = model_fn(**params)
        
        # 添加剪枝回调
        callbacks = [
            TFKerasPruningCallback(trial, metric)
        ]
        
        # 训练模型
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            callbacks=callbacks,
            verbose=0
        )
        
        # 返回优化指标
        return history.history[metric][-1]
    
    # 创建Optuna研究
    study = optuna.create_study(direction=direction, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    
    print(f"最佳参数: {study.best_params}")
    print(f"最佳{metric}: {study.best_value}")
    
    return study.best_params

def optimize_with_ray_tune(train_fn: Callable,
                         param_space: Dict[str, Any],
                         num_samples: int = 10,
                         resources_per_trial: Dict[str, float] = {"cpu": 1, "gpu": 0.5},
                         metric: str = "val_accuracy",
                         mode: str = "max") -> Dict[str, Any]:
    """使用Ray Tune进行超参数优化。
    
    Args:
        train_fn: 训练函数
        param_space: 参数空间
        num_samples: 采样次数
        resources_per_trial: 每个试验的资源
        metric: 优化指标
        mode: 优化模式，'max'或'min'
        
    Returns:
        最佳参数
    """
    # 初始化Ray
    if not ray.is_initialized():
        ray.init()
    
    # 创建调度器
    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=10,  # 最大轮数
        grace_period=1,  # 最小轮数
        reduction_factor=2
    )
    
    # 运行优化
    result = tune.run(
        train_fn,
        config=param_space,
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial=resources_per_trial,
        progress_reporter=tune.CLIReporter(
            metric_columns=["loss", metric, "training_iteration"]
        )
    )
    
    # 获取最佳配置
    best_trial = result.get_best_trial(metric=metric, mode=mode)
    best_config = best_trial.config
    best_result = best_trial.last_result[metric]
    
    print(f"最佳参数: {best_config}")
    print(f"最佳{metric}: {best_result}")
    
    return best_config
```

### 模型压缩与量化

在`pyproject.toml`中添加依赖：

```toml
[project.optional-dependencies]
model_compression = [
    "torch>=1.10.0",
    "tensorflow>=2.8.0",
    "onnx>=1.10.0",
    "onnxruntime>=1.9.0",
    "tensorflow-model-optimization>=0.7.0",
]
```

创建模型压缩与量化模块：

```python
# src/your_package_name/optimization/model_compression.py
import torch
import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import os

# PyTorch模型量化
def quantize_pytorch_model(model: torch.nn.Module, 
                         example_input: torch.Tensor,
                         quantization_type: str = 'dynamic') -> torch.nn.Module:
    """量化PyTorch模型。
    
    Args:
        model: 要量化的模型
        example_input: 示例输入
        quantization_type: 量化类型，'dynamic'或'static'
        
    Returns:
        量化后的模型
    """
    model.eval()
    
    if quantization_type == 'dynamic':
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model,  # 原始模型
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},  # 要量化的层类型
            dtype=torch.qint8  # 量化数据类型
        )
    elif quantization_type == 'static':
        # 静态量化
        # 1. 设置量化配置
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 2. 准备量化
        model_prepared = torch.quantization.prepare(model)
        
        # 3. 校准（需要使用代表性数据集）
        # 这里简单示例，实际应用中应该使用校准数据集
        model_prepared(example_input)
        
        # 4. 转换为量化模型
        quantized_model = torch.quantization.convert(model_prepared)
    
    return quantized_model

# 模型剪枝
def prune_pytorch_model(model: torch.nn.Module, 
                      pruning_method: str = 'l1_unstructured',
                      amount: float = 0.2) -> torch.nn.Module:
    """剪枝PyTorch模型。
    
    Args:
        model: 要剪枝的模型
        pruning_method: 剪枝方法，'l1_unstructured'或'random_unstructured'
        amount: 剪枝比例
        
    Returns:
        剪枝后的模型
    """
    import torch.nn.utils.prune as prune
    
    # 获取所有卷积层和线性层
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if pruning_method == 'l1_unstructured':
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif pruning_method == 'random_unstructured':
                prune.random_unstructured(module, name='weight', amount=amount)
            
            # 使剪枝永久化
            prune.remove(module, 'weight')
    
    return model

# TensorFlow模型量化
def quantize_tensorflow_model(model: tf.keras.Model,
                           dataset: tf.data.Dataset,
                           quantization_type: str = 'post_training') -> tf.keras.Model:
    """量化TensorFlow模型。
    
    Args:
        model: 要量化的模型
        dataset: 代表性数据集，用于校准
        quantization_type: 量化类型，'post_training'或'aware_training'
        
    Returns:
        量化后的模型
    """
    import tensorflow_model_optimization as tfmot
    
    if quantization_type == 'post_training':
        # 训练后量化
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 代表性数据集用于校准
        def representative_dataset_gen():
            for data, _ in dataset.take(100).as_numpy_iterator():
                yield [data]
        
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        quantized_tflite_model = converter.convert()
        
        # 保存量化模型
        with open('quantized_model.tflite', 'wb') as f:
            f.write(quantized_tflite_model)
        
        # 加载量化模型
        interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        
        return interpreter
    
    elif quantization_type == 'aware_training':
        # 量化感知训练
        quantize_model = tfmot.quantization.keras.quantize_model
        
        # 创建量化感知模型
        q_aware_model = quantize_model(model)
        
        # 编译模型
        q_aware_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 训练量化感知模型
        q_aware_model.fit(dataset, epochs=1)  # 实际应用中应该训练更多轮
        
        # 转换为TFLite模型
        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()
        
        # 保存量化模型
        with open('quantized_model.tflite', 'wb') as f:
            f.write(quantized_tflite_model)
        
        return q_aware_model

# 模型蒸馏
def knowledge_distillation(teacher_model: torch.nn.Module,
                         student_model: torch.nn.Module,
                         train_loader: torch.utils.data.DataLoader,
                         val_loader: torch.utils.data.DataLoader,
                         temperature: float = 5.0,
                         alpha: float = 0.5,
                         epochs: int = 10,
                         lr: float = 0.001) -> torch.nn.Module:
    """知识蒸馏训练学生模型。
    
    Args:
        teacher_model: 教师模型
        student_model: 学生模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        temperature: 温度参数
        alpha: 蒸馏损失权重
        epochs: 训练轮数
        lr: 学习率
        
    Returns:
        训练好的学生模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # 设置教师模型为评估模式
    teacher_model.eval()
    
    # 定义优化器
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    
    # 定义损失函数
    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(epochs):
        student_model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 教师模型前向传播
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            # 学生模型前向传播
            student_logits = student_model(inputs)
            
            # 计算蒸馏损失
            # 1. 软目标损失
            soft_targets = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
            soft_prob = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
            distillation_loss = kl_loss(soft_prob, soft_targets) * (temperature ** 2)
            
            # 2. 硬目标损失
            student_loss = ce_loss(student_logits, targets)
            
            # 总损失
            loss = alpha * distillation_loss + (1 - alpha) * student_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student_model(inputs)
                loss = ce_loss(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return student_model
```

## 模型部署与服务

### 模型序列化

在`pyproject.toml`中添加依赖：

```toml
[project.optional-dependencies]
model_serving = [
    "onnx>=1.10.0",
    "onnxruntime>=1.9.0",
    "tensorflowjs>=3.15.0",
    "torch>=1.10.0",
    "tensorflow>=2.8.0",
]
```

创建模型序列化模块：

```python
# src/your_package_name/deployment/model_serialization.py
import torch
import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import os
import json

def export_pytorch_to_onnx(model: torch.nn.Module,
                          example_input: torch.Tensor,
                          output_path: str,
                          input_names: List[str] = ['input'],
                          output_names: List[str] = ['output'],
                          dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None) -> str:
    """将PyTorch模型导出为ONNX格式。
    
    Args:
        model: PyTorch模型
        example_input: 示例输入张量
        output_path: 输出路径
        input_names: 输入名称列表
        output_names: 输出名称列表
        dynamic_axes: 动态轴配置
        
    Returns:
        ONNX模型路径
    """
    model.eval()
    
    # 设置动态轴（批次维度）
    if dynamic_axes is None:
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    # 导出模型
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # 验证ONNX模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"PyTorch模型已导出为ONNX格式: {output_path}")
    return output_path

def export_tensorflow_to_savedmodel(model: tf.keras.Model,
                                  output_dir: str) -> str:
    """将TensorFlow模型导出为SavedModel格式。
    
    Args:
        model: TensorFlow模型
        output_dir: 输出目录
        
    Returns:
        SavedModel目录路径
    """
    # 导出模型
    tf.saved_model.save(model, output_dir)
    
    print(f"TensorFlow模型已导出为SavedModel格式: {output_dir}")
    return output_dir

def export_tensorflow_to_tfjs(model: tf.keras.Model,
                            output_dir: str) -> str:
    """将TensorFlow模型导出为TensorFlow.js格式。
    
    Args:
        model: TensorFlow模型
        output_dir: 输出目录
        
    Returns:
        TensorFlow.js模型目录路径
    """
    import tensorflowjs as tfjs
    
    # 导出模型
    tfjs.converters.save_keras_model(model, output_dir)
    
    print(f"TensorFlow模型已导出为TensorFlow.js格式: {output_dir}")
    return output_dir

def export_tensorflow_to_tflite(model: tf.keras.Model,
                              output_path: str,
                              optimize: bool = True) -> str:
    """将TensorFlow模型导出为TFLite格式。
    
    Args:
        model: TensorFlow模型
        output_path: 输出路径
        optimize: 是否优化模型
        
    Returns:
        TFLite模型路径
    """
    # 创建转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 设置优化
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 转换模型
    tflite_model = converter.convert()
    
    # 保存模型
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow模型已导出为TFLite格式: {output_path}")
    return output_path

def export_model_metadata(model_info: Dict[str, Any],
                        output_path: str) -> str:
    """导出模型元数据。
    
    Args:
        model_info: 模型信息字典，包含模型名称、版本、输入输出规范等
        output_path: 输出路径
        
    Returns:
        元数据文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入元数据
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"模型元数据已导出: {output_path}")
    return output_path

def load_onnx_model(model_path: str) -> ort.InferenceSession:
    """加载ONNX模型。
    
    Args:
        model_path: ONNX模型路径
        
    Returns:
        ONNX推理会话
    """
    # 创建推理会话
    session = ort.InferenceSession(model_path)
    
    print(f"ONNX模型已加载: {model_path}")
    return session

def onnx_inference(session: ort.InferenceSession,
                 input_data: np.ndarray,
                 input_name: Optional[str] = None) -> np.ndarray:
    """使用ONNX模型进行推理。
    
    Args:
        session: ONNX推理会话
        input_data: 输入数据
        input_name: 输入名称，如果为None则使用模型的第一个输入
        
    Returns:
        推理结果
    """
    # 获取输入名称
    if input_name is None:
        input_name = session.get_inputs()[0].name
    
    # 获取输出名称
    output_name = session.get_outputs()[0].name
    
    # 进行推理
    results = session.run([output_name], {input_name: input_data})
    
    return results[0]
```

### REST API服务

在`pyproject.toml`中添加依赖：

```toml
[project.optional-dependencies]
api_serving = [
    "fastapi>=0.75.0",
    "uvicorn>=0.17.0",
    "pydantic>=1.9.0",
    "python-multipart>=0.0.5",
]
```

创建REST API服务模块：

```python
# src/your_package_name/deployment/rest_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import torch
import tensorflow as tf
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import io
import time
import os
import json
from PIL import Image

# 定义请求和响应模型
class PredictionRequest(BaseModel):
    """预测请求模型。"""
    inputs: List[List[float]]
    
class PredictionResponse(BaseModel):
    """预测响应模型。"""
    predictions: List[List[float]]
    model_version: str
    processing_time: float

class ModelService:
    """模型服务类。
    
    Args:
        model_path: 模型路径
        model_type: 模型类型，'onnx', 'pytorch'或'tensorflow'
        metadata_path: 元数据路径
    """
    def __init__(self, model_path: str, model_type: str, metadata_path: Optional[str] = None):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.model = self._load_model()
        self.metadata = self._load_metadata(metadata_path) if metadata_path else {}
        self.version = self.metadata.get('version', '1.0.0')
    
    def _load_model(self) -> Any:
        """加载模型。
        
        Returns:
            加载的模型
        """
        if self.model_type == 'onnx':
            return ort.InferenceSession(self.model_path)
        elif self.model_type == 'pytorch':
            model = torch.load(self.model_path, map_location=torch.device('cpu'))
            model.eval()
            return model
        elif self.model_type == 'tensorflow':
            return tf.saved_model.load(self.model_path)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """加载元数据。
        
        Args:
            metadata_path: 元数据路径
            
        Returns:
            元数据字典
        """
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """进行预测。
        
        Args:
            inputs: 输入数据
            
        Returns:
            预测结果
        """
        start_time = time.time()
        
        if self.model_type == 'onnx':
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            results = self.model.run([output_name], {input_name: inputs})[0]
        
        elif self.model_type == 'pytorch':
            with torch.no_grad():
                inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
                outputs = self.model(inputs_tensor)
                results = outputs.numpy()
        
        elif self.model_type == 'tensorflow':
            results = self.model(tf.constant(inputs, dtype=tf.float32)).numpy()
        
        self.last_processing_time = time.time() - start_time
        return results

# 创建FastAPI应用
def create_model_api(model_service: ModelService) -> FastAPI:
    """创建模型API。
    
    Args:
        model_service: 模型服务实例
        
    Returns:
        FastAPI应用
    """
    app = FastAPI(title="模型服务API", description="机器学习模型REST API服务")
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """进行预测。
        
        Args:
            request: 预测请求
            
        Returns:
            预测响应
        """
        try:
            inputs = np.array(request.inputs, dtype=np.float32)
            predictions = model_service.predict(inputs)
            
            return PredictionResponse(
                predictions=predictions.tolist(),
                model_version=model_service.version,
                processing_time=model_service.last_processing_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict_image")
    async def predict_image(file: UploadFile = File(...)):
        """预测图像。
        
        Args:
            file: 上传的图像文件
            
        Returns:
            预测响应
        """
        try:
            # 读取图像
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # 预处理图像（示例）
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # 预测
            predictions = model_service.predict(image_array)
            
            return JSONResponse(content={
                "predictions": predictions.tolist(),
                "model_version": model_service.version,
                "processing_time": model_service.last_processing_time
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """健康检查。
        
        Returns:
            健康状态
        """
        return {"status": "healthy", "model_version": model_service.version}
    
    @app.get("/metadata")
    async def get_metadata():
        """获取模型元数据。
        
        Returns:
            模型元数据
        """
        return model_service.metadata
    
    return app

def run_model_server(model_path: str, 
                    model_type: str, 
                    metadata_path: Optional[str] = None,
                    host: str = "0.0.0.0", 
                    port: int = 8000):
    """运行模型服务器。
    
    Args:
        model_path: 模型路径
        model_type: 模型类型
        metadata_path: 元数据路径
        host: 主机地址
        port: 端口号
    """
    # 创建模型服务
    model_service = ModelService(model_path, model_type, metadata_path)
    
    # 创建API
    app = create_model_api(model_service)
    
    # 运行服务器
    uvicorn.run(app, host=host, port=port)
```

### 边缘设备部署

在`pyproject.toml`中添加依赖：

```toml
[project.optional-dependencies]
edge_deployment = [
    "tflite-runtime>=2.5.0",
    "onnxruntime>=1.9.0",
    "opencv-python>=4.5.0",
]
```

创建边缘设备部署模块：

```python
# src/your_package_name/deployment/edge_deployment.py
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import os
import json
import time

class EdgeModelWrapper:
    """边缘设备模型包装器。
    
    Args:
        model_path: 模型路径
        model_type: 模型类型，'tflite'或'onnx'
        metadata_path: 元数据路径
    """
    def __init__(self, model_path: str, model_type: str, metadata_path: Optional[str] = None):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.model = self._load_model()
        self.metadata = self._load_metadata(metadata_path) if metadata_path else {}
    
    def _load_model(self) -> Any:
        """加载模型。
        
        Returns:
            加载的模型
        """
        if self.model_type == 'tflite':
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            
            interpreter = tflite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        
        elif self.model_type == 'onnx':
            import onnxruntime as ort
            return ort.InferenceSession(self.model_path)
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """加载元数据。
        
        Args:
            metadata_path: 元数据路径
            
        Returns:
            元数据字典
        """
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """进行预测。
        
        Args:
            inputs: 输入数据
            
        Returns:
            预测结果
        """
        start_time = time.time()
        
        if self.model_type == 'tflite':
            # 获取输入输出细节
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            
            # 设置输入张量
            self.model.set_tensor(input_details[0]['index'], inputs)
            
            # 运行推理
            self.model.invoke()
            
            # 获取输出张量
            results = self.model.get_tensor(output_details[0]['index'])
        
        elif self.model_type == 'onnx':
            # 获取输入名称
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            # 运行推理
            results = self.model.run([output_name], {input_name: inputs})[0]
        
        self.last_processing_time = time.time() - start_time
        return results

def process_camera_feed(model_wrapper: EdgeModelWrapper, 
                       camera_id: int = 0,
                       preprocessing_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                       postprocessing_fn: Optional[Callable[[np.ndarray], Any]] = None,
                       display_results: bool = True) -> None:
    """处理摄像头视频流。
    
    Args:
        model_wrapper: 边缘模型包装器
        camera_id: 摄像头ID
        preprocessing_fn: 预处理函数
        postprocessing_fn: 后处理函数
        display_results: 是否显示结果
    """
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break
            
            # 预处理
            if preprocessing_fn is not None:
                processed_frame = preprocessing_fn(frame)
            else:
                # 默认预处理：调整大小并归一化
                processed_frame = cv2.resize(frame, (224, 224))
                processed_frame = processed_frame.astype(np.float32) / 255.0
                processed_frame = np.expand_dims(processed_frame, axis=0)
            
            # 推理
            results = model_wrapper.predict(processed_frame)
            
            # 后处理
            if postprocessing_fn is not None:
                processed_results = postprocessing_fn(results)
            else:
                processed_results = results
            
            # 显示结果
            if display_results:
                # 在帧上绘制结果（示例）
                if isinstance(processed_results, np.ndarray) and processed_results.size > 0:
                    # 假设结果是分类概率
                    top_class = np.argmax(processed_results[0])
                    confidence = processed_results[0][top_class]
                    cv2.putText(frame, f"Class: {top_class}, Conf: {confidence:.2f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow('Edge AI Demo', frame)
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        # 释放资源
        cap.release()
        if display_results:
            cv2.destroyAllWindows()

def deploy_model_to_edge(model_path: str,
                        output_path: str,
                        model_type: str = 'tflite',
                        optimization_level: int = 3,
                        target_device: str = 'generic') -> str:
    """将模型部署到边缘设备。
    
    Args:
        model_path: 模型路径
        output_path: 输出路径
        model_type: 模型类型，'tflite'或'onnx'
        optimization_level: 优化级别
        target_device: 目标设备
        
    Returns:
        优化后的模型路径
    """
    if model_type == 'tflite':
        import tensorflow as tf
        
        # 加载模型
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path) \
            if os.path.isdir(model_path) else tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(model_path))
        
        # 设置优化选项
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 根据目标设备设置委托
        if target_device == 'gpu':
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        elif target_device == 'dsp' or target_device == 'npu':
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # 转换模型
        tflite_model = converter.convert()
        
        # 保存模型
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        return output_path
    
    elif model_type == 'onnx':
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # 加载模型
        model = onnx.load(model_path)
        
        # 优化模型
        if optimization_level > 0:
            # 量化模型
            quantized_model_path = output_path.replace('.onnx', '_quantized.onnx')
            quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QUInt8)
            return quantized_model_path
        else:
            # 直接复制模型
            onnx.save(model, output_path)
            return output_path
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
```

## 常见问题与解决方案

### 内存管理问题

在处理大型模型和数据集时，内存管理是一个常见挑战。以下是一些常见问题和解决方案：

#### 问题：训练时内存溢出（OOM）

**症状**：训练过程中出现`RuntimeError: CUDA out of memory`或`MemoryError`。

**解决方案**：

1. **减小批次大小**：降低每个批次的样本数量。

```python
# 减小批次大小
train_loader = DataLoader(train_dataset, batch_size=32)  # 原始批次大小
train_loader = DataLoader(train_dataset, batch_size=16)  # 减小的批次大小
```

2. **使用梯度累积**：在多个小批次上累积梯度，然后一次性更新模型。

```python
# PyTorch中的梯度累积
accumulation_steps = 4  # 累积4个批次的梯度
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps  # 缩放损失
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **启用混合精度训练**：使用较低精度（如float16）进行部分计算。

```python
# PyTorch中的混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

4. **使用模型并行或分布式训练**：将模型分割到多个GPU上。

```python
# 使用DataParallel进行简单的模型并行
model = nn.DataParallel(model)
```

5. **优化数据加载**：使用内存映射文件或数据生成器。

```python
# 使用NumPy的内存映射文件
import numpy as np

# 创建内存映射文件
data = np.memmap('large_data.dat', dtype='float32', mode='w+', shape=(10000, 1000))

# 使用内存映射文件
for i in range(10000):
    # 处理数据的一部分
    chunk = data[i:i+100]
```

#### 问题：推理时内存泄漏

**症状**：长时间运行的服务内存使用量持续增加。

**解决方案**：

1. **定期清理缓存**：在PyTorch中清除CUDA缓存。

```python
import torch
import gc

def inference_with_cleanup(model, inputs, cleanup_every=100):
    results = []
    for i, input_batch in enumerate(inputs):
        with torch.no_grad():
            output = model(input_batch)
            results.append(output.cpu().numpy())  # 将结果移动到CPU
        
        # 定期清理
        if (i + 1) % cleanup_every == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return results
```

2. **使用上下文管理器**：确保资源正确释放。

```python
class ModelContext:
    def __init__(self, model):
        self.model = model
    
    def __enter__(self):
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清理资源
        torch.cuda.empty_cache()
        gc.collect()

# 使用上下文管理器
with ModelContext(model) as m:
    output = m(input_data)
```

3. **检查张量引用**：确保不保留不必要的张量引用。

```python
# 不好的做法：保留中间结果
intermediate_results = []
for data in dataloader:
    output = model(data)
    intermediate_results.append(output)  # 保留引用

# 好的做法：只保留必要的结果
final_results = []
for data in dataloader:
    output = model(data)
    processed_result = process_output(output)  # 处理输出
    final_results.append(processed_result)  # 只保留处理后的结果
```

### GPU相关问题

#### 问题：GPU利用率低

**症状**：训练过程中GPU利用率远低于100%。

**解决方案**：

1. **增加批次大小**：如果内存允许，增加批次大小可以提高GPU利用率。

```python
# 增加批次大小
train_loader = DataLoader(train_dataset, batch_size=64)  # 原始批次大小
train_loader = DataLoader(train_dataset, batch_size=128)  # 增加的批次大小
```

2. **优化数据加载**：使用多个工作进程和预取。

```python
# 优化数据加载
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=4,  # 使用多个工作进程
    pin_memory=True,  # 将数据固定在内存中
    prefetch_factor=2  # 预取因子
)
```

3. **使用更高效的模型架构**：某些操作可能导致GPU利用率低。

```python
# 使用更高效的卷积操作
# 替换
# self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
# 为
self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1)
```

4. **检查是否存在CPU瓶颈**：确保数据预处理不是瓶颈。

```python
# 将数据预处理移到GPU上
def preprocess_on_gpu(data):
    data = data.to('cuda')  # 先移动到GPU
    # 在GPU上进行预处理
    data = data / 255.0
    return data
```

#### 问题：多GPU训练效率低

**症状**：使用多个GPU训练时，加速比远低于GPU数量。

**解决方案**：

1. **使用更高效的分布式训练方法**：从`DataParallel`升级到`DistributedDataParallel`。

```python
# 使用DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')

# 创建DDP模型
model = model.to(device_id)
model = DDP(model, device_ids=[device_id])
```

2. **优化批次大小和学习率**：根据GPU数量调整超参数。

```python
# 根据GPU数量调整批次大小和学习率
num_gpus = torch.cuda.device_count()
batch_size = 32 * num_gpus  # 线性缩放批次大小
learning_rate = 0.001 * num_gpus  # 线性缩放学习率
```

3. **减少GPU之间的通信**：使用梯度累积减少同步频率。

```python
# 使用梯度累积减少同步频率
accumulation_steps = 4  # 累积4个批次的梯度
for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 依赖冲突问题

#### 问题：框架版本冲突

**症状**：安装多个机器学习框架时出现依赖冲突，例如TensorFlow和PyTorch依赖不同版本的NumPy。

**解决方案**：

1. **使用虚拟环境**：为不同的项目创建独立的虚拟环境。

```bash
# 使用conda创建虚拟环境
conda create -n torch_env python=3.9
conda activate torch_env
pip install torch torchvision

# 为另一个项目创建不同的环境
conda create -n tf_env python=3.9
conda activate tf_env
pip install tensorflow
```

2. **使用可选依赖**：在`pyproject.toml`中使用可选依赖分组。

```toml
[project.optional-dependencies]
torch = [
    "torch>=1.10.0",
    "torchvision>=0.11.0",
]
tensorflow = [
    "tensorflow>=2.8.0",
]
```

3. **指定兼容版本范围**：明确指定依赖的版本范围。

```toml
[project.dependencies]
numpy = ">=1.20.0,<1.24.0"  # 指定兼容的NumPy版本范围
```

4. **使用依赖解析工具**：使用`pip-tools`或`poetry`管理依赖。

```bash
# 使用pip-tools
pip install pip-tools
pip-compile requirements.in  # 生成固定版本的requirements.txt
pip-sync requirements.txt   # 安装精确的依赖版本
```

#### 问题：CUDA版本冲突

**症状**：不同的深度学习框架需要不同版本的CUDA。

**解决方案**：

1. **使用容器化技术**：使用Docker隔离不同的CUDA环境。

```bash
# 使用特定CUDA版本的PyTorch容器
docker pull pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
docker run -it --gpus all pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
```

2. **安装特定CUDA版本的框架**：选择与系统CUDA版本兼容的框架版本。

```bash
# 安装与CUDA 11.3兼容的PyTorch
pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

3. **使用CPU版本**：如果CUDA版本冲突无法解决，可以使用CPU版本。

```bash
# 安装CPU版本的PyTorch
pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

#### 问题：自定义操作编译问题

**症状**：编译自定义操作（如TensorFlow的自定义算子或PyTorch的C++扩展）时出现错误。

**解决方案**：

1. **确保编译环境一致**：使用与框架兼容的编译器版本。

```bash
# 检查TensorFlow兼容的编译器版本
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# 安装特定版本的GCC
sudo apt-get install gcc-7 g++-7
export CC=/usr/bin/gcc-7
export CXX=/usr/bin/g++-7
```

2. **使用预编译的扩展**：尽可能使用预编译的扩展包。

```python
# 在setup.py中使用预编译的扩展
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_ops",
    ext_modules=[
        CUDAExtension(
            name="custom_cuda",
            sources=["custom_op.cpp", "custom_op_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)
```

3. **使用JIT编译**：使用即时编译避免预编译问题。

```python
# PyTorch中使用JIT编译
from torch.utils.cpp_extension import load

custom_op = load(
    name="custom_op",
    sources=["custom_op.cpp", "custom_op_kernel.cu"],
    verbose=True
)
```

#### 问题：分布式训练环境冲突

**症状**：在分布式环境中，不同节点的环境配置不一致导致训练失败。

**解决方案**：

1. **使用容器化技术**：确保所有节点使用相同的容器镜像。

```bash
# 在所有节点上拉取相同的Docker镜像
docker pull nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
```

2. **创建环境配置脚本**：使用脚本确保所有节点安装相同的依赖。

```bash
#!/bin/bash
# setup_env.sh

# 创建虚拟环境
python -m venv env
source env/bin/activate

# 安装固定版本的依赖
pip install -r requirements.txt

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
```

3. **使用环境管理工具**：使用Conda或Poetry管理环境。

```bash
# 使用Conda环境文件
conda env create -f environment.yml
```

```yaml
# environment.yml
name: ml_project
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.9
  - pytorch=1.12.0
  - cudatoolkit=11.3
  - numpy=1.22.0
  - pip:
    - tensorflow==2.8.0
```

通过以上解决方案，您可以有效地处理机器学习和深度学习项目中常见的依赖冲突问题，确保项目能够顺利开发和部署。{