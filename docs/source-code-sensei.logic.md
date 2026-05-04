# openpi 流程图

## 总体架构

```mermaid
flowchart TD
    User["用户/机器人环境/训练脚本"] --> Config["training.config<br/>TrainConfig / DataConfig"]
    Config --> DataLoader["training.data_loader<br/>LeRobot / RLDS / Fake"]
    DataLoader --> Transforms["openpi.transforms<br/>repack / normalize / tokenize"]
    Transforms --> Observation["models.model.Observation"]
    Observation --> Model["models.pi0 / pi0_fast / models_pytorch"]
    Model --> Actions["动作 chunk / action tokens"]
    Actions --> Policy["policies.policy<br/>输入输出适配"]
    Policy --> Server["serving.websocket_policy_server"]
    Server --> Client["openpi-client.websocket_client_policy"]
    Client --> User
```

## 数据生命周期

```mermaid
flowchart LR
    Raw["原始样本<br/>LeRobot / RLDS / 环境 obs"] --> Repack["RepackTransform"]
    Repack --> RobotTransform["机器人专用 transform<br/>ALOHA / DROID / LIBERO"]
    RobotTransform --> Norm["Normalize<br/>z-score / quantile"]
    Norm --> ModelTransform["模型 transform<br/>resize / tokenize / pad"]
    ModelTransform --> Dict["统一 dict<br/>image / image_mask / state / tokens / actions"]
    Dict --> Obs["Observation.from_dict"]
    Obs --> Model["compute_loss / sample_actions"]
    Model --> OutTransform["输出反变换<br/>decode / unnormalize / robot outputs"]
    OutTransform --> Action["机器人动作"]
```

## 训练调用链

```mermaid
sequenceDiagram
    participant CLI as scripts/train.py
    participant Config as training.config
    participant Loader as training.data_loader
    participant Model as models.BaseModel
    participant Opt as training.optimizer
    participant Ckpt as training.checkpoints

    CLI->>Config: cli() 解析 TrainConfig
    CLI->>Loader: create_data_loader(config)
    Loader->>Config: data.create(...)
    Loader->>Loader: 创建 dataset 并应用 transforms
    CLI->>Model: config.model.create(init_rng)
    CLI->>Model: weight_loader 加载预训练参数
    CLI->>Opt: create_optimizer(...)
    loop train step
        Loader-->>CLI: Observation, Actions
        CLI->>Model: compute_loss(...)
        CLI->>Opt: value_and_grad + update
        CLI->>Ckpt: save_state(...)
    end
```

## 推理调用链

```mermaid
sequenceDiagram
    participant App as 用户代码/示例
    participant PC as policy_config
    participant Policy as Policy
    participant TF as transforms
    participant Model as JAX/PyTorch Model

    App->>PC: create_trained_policy(config, checkpoint)
    PC->>Model: load params 或 model.safetensors
    PC->>TF: 组装输入/输出 transforms
    PC-->>App: Policy
    App->>Policy: infer(raw_obs)
    Policy->>TF: 输入转换 + normalize + tokenize
    Policy->>Model: sample_actions(...)
    Model-->>Policy: normalized actions
    Policy->>TF: decode + unnormalize + 输出转换
    Policy-->>App: actions + timing
```

## 远程推理

```mermaid
flowchart TD
    Serve["scripts/serve_policy.py"] --> CreatePolicy["policy_config.create_trained_policy"]
    CreatePolicy --> WSServer["WebsocketPolicyServer"]
    Robot["机器人端 example/runtime"] --> WSClient["WebsocketClientPolicy"]
    WSClient -->|msgpack obs| WSServer
    WSServer -->|Policy.infer| Policy["Policy"]
    Policy -->|actions| WSServer
    WSServer -->|msgpack actions| WSClient
    WSClient --> Broker["ActionChunkBroker"]
    Broker --> Env["Environment.step"]
```

## pi0 / pi05 Flow Matching

```mermaid
flowchart TD
    Images["多路 RGB 图像"] --> SigLIP["SigLIP 图像编码"]
    Prompt["语言 prompt"] --> Tokenizer["PaliGemma tokenizer"]
    Tokenizer --> LLMEmbed["Gemma token embedding"]
    State["机器人 state"] --> StateProj["state/action expert 投影"]
    Actions["真实 actions"] --> NoiseMix["采样 time 和 noise<br/>构造 x_t 与 u_t"]
    NoiseMix --> Suffix["动作 suffix tokens"]
    SigLIP --> Prefix["prefix tokens"]
    LLMEmbed --> Prefix
    StateProj --> Suffix
    Prefix --> Mask["attention mask / positions"]
    Suffix --> Mask
    Mask --> PaliGemma["PaliGemma LLM"]
    PaliGemma --> OutProj["action_out_proj"]
    OutProj --> Loss["v_t 对齐 u_t<br/>MSE loss"]
```

## pi0-FAST

```mermaid
flowchart TD
    PromptStateAction["prompt + state + actions"] --> FASTTok["FASTTokenizer"]
    FASTTok --> Tokens["自回归 token 序列"]
    Images["图像 tokens"] --> Prefix["视觉/语言 prefix"]
    Tokens --> AR["autoregressive suffix"]
    Prefix --> GemmaFast["Gemma FAST"]
    AR --> GemmaFast
    GemmaFast --> TokenLogits["action token logits"]
    TokenLogits --> Extract["ExtractFASTActions"]
    Extract --> Actions["连续动作 chunk"]
```

## 配置系统

```mermaid
flowchart TD
    TrainConfig["TrainConfig"] --> ModelConfig["model: BaseModelConfig"]
    TrainConfig --> DataFactory["data: DataConfigFactory"]
    TrainConfig --> WeightLoader["weight_loader"]
    TrainConfig --> Optimizer["optimizer + lr_schedule"]
    DataFactory --> DataConfig["DataConfig"]
    DataConfig --> Repack["repack_transforms"]
    DataConfig --> RobotTF["data_transforms"]
    DataConfig --> ModelTF["model_transforms"]
    DataConfig --> Norm["norm_stats / asset_id"]
```
