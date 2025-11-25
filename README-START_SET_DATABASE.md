# Start Set Database - README

## Objective

The objective of the Start Set Database is to provide a set of pre-trained machines that enable efficient neural network (NN) training. These machines contain accumulated knowledge and experiences from previous training runs, allowing the system to make informed decisions about:

- Optimal neural network architectures and configurations
- Effective feature engineering transformer (FET) selections
- Best practices learned from various datasets and training scenarios

By leveraging these trained machines, the system can significantly reduce the time and computational resources required to train new neural networks, as it can start from proven configurations rather than exploring the entire search space from scratch.

## How to Get the Database (via Git LFS)

This project uses Git LFS (Large File Storage) to store large files such as the database (.sqlite3).

If you just clone the repo normally, you'll only get small pointer files — not the real database.

To get the full database, follow these steps:

### 🧩 1. Install Git LFS

**On Windows:**

Download and install from:

👉 https://git-lfs.github.com/

Then open Git Bash and run:

```bash
git lfs install
```

**On macOS:**

```bash
brew install git-lfs
git lfs install
```

**On Linux (Debian/Ubuntu):**

```bash
sudo apt install git-lfs
git lfs install
```

### 🧭 2. Clone the Repository

Clone as you normally would:

```bash
git clone https://github.com/EasyAutoML-com/EasyAutoML-Core.git
```

Git LFS will automatically download the actual database files during the clone process.

## Indicative List of Machines

The following is an indicative list of machine types that may be present in the database. **Note: The actual list of machines can vary over time** as new experiences are accumulated and the system evolves.

### 1. Neural Network Configuration Machines

These machines store experiences and results about neural network architectures:

- **`__SF_Experiences_NNConfiguration--Level=1--Inputs=<100--Outputs=<100__`**
  - Stores experiences for Level 1 networks with inputs < 100 and outputs < 100

- **`__SF_Experiences_NNConfiguration--Level=1--Inputs=100-1000--Outputs=<100__`**
  - Stores experiences for Level 1 networks with inputs between 100-1000 and outputs < 100

- **`__SF_Experiences_NNConfiguration--Level=1--Inputs=<100--Outputs=>100__`**
  - Stores experiences for Level 1 networks with inputs < 100 and outputs > 100

- **`__SF_Experiences_NNConfiguration--Level=1--Inputs=100-1000--Outputs=>100__`**
  - Stores experiences for Level 1 networks with inputs between 100-1000 and outputs > 100

- **`__SF_Experiences_NNConfiguration--Level=2--Inputs=<100--Outputs=<100__`**
  - Stores experiences for Level 2 networks with inputs < 100 and outputs < 100

- **`__SF_Experiences_NNConfiguration--Level=2--Inputs=100-1000--Outputs=<100__`**
  - Stores experiences for Level 2 networks with inputs between 100-1000 and outputs < 100

- **`__SF_Experiences_NNConfiguration--Level=3--Inputs=<100--Outputs=<100__`**
  - Stores experiences for Level 3 networks with inputs < 100 and outputs < 100

- **`__SF_Experiences_NNConfiguration--Level=3--Inputs=100-1000--Outputs=<100__`**
  - Stores experiences for Level 3 networks with inputs between 100-1000 and outputs < 100

- **`__SF_Results_NNConfiguration--Level=1--Inputs=<100--Outputs=<100__`**
  - Stores results for Level 1 networks with inputs < 100 and outputs < 100

- **`__SF_Results_NNConfiguration--Level=2--Inputs=<100--Outputs=<100__`**
  - Stores results for Level 2 networks with inputs < 100 and outputs < 100

### 2. Column Feature Engineering Transformer (FET) Selector Machines

These machines store experiences and results about feature engineering transformer selections with different budgets:

- **Budget 16 Machines:**
  - Various combinations of FETs including:
    - `FET6PowerFloat`, `FETMultiplexerAllFloat`, `FETMultiplexerMostFrequentsValuesFloat`
    - `FETNumericMinMaxFloat`, `FETNumericPowerTransformerFloat`
    - `FETNumericQuantileTransformerFloat`, `FETNumericQuantileTransformerNormalFloat`
    - `FETNumericRobustScalerFloat`, `FETNumericStandardFloat`
  - Similar combinations for `Label` and `Date/Time` data types

- **Budget 64 Machines:**
  - Similar FET combinations but with higher budget constraints
  - Includes both `Float`, `Label`, and `Date/Time` variants

- **Budget 256 Machines:**
  - Similar FET combinations with the highest budget constraints
  - Includes both `Float` and `Label` variants

**Example machine names:**
- `__SF_Experiences_ColumnFETSelector--Budget=16--FETs=(FETNumericMinMaxFloat+FETNumericPowerTransformerFloat+...)__`
- `__SF_Results_ColumnFETSelector--Budget=16--FETs=(FET6PowerFloat+FETMultiplexerMostFrequentsValuesFloat+...)__`

### 3. Training Results Machines

These machines store aggregated results from training processes:

- **`__Results_Find_Best_NN_Configuration__`**
  - Stores results from searches for the best neural network configuration

- **`__Results_NNEngine_Training_Cycle__`**
  - Stores results from individual training cycles

- **`__Results_NNEngine_Training_Full__`**
  - Stores comprehensive results from full training runs

### 4. Test Dataset Machines

These machines are created from various test datasets to validate the system:

- **Large Dataset Machines:**
  - `TEST MACHINE DATASET:(Large) bank-marketing-full.xls`
  - `TEST MACHINE DATASET:(Large) chatgpt 75000 tweets`
  - `TEST MACHINE DATASET:(Large) diabetic 100000 data - Plus 1750 NO to predict`
  - `TEST MACHINE DATASET:(Large) googleplaystore_user_reviews`
  - `TEST MACHINE DATASET:(Large) Predict next-day rain by training classification models on the target variable RainTomorrow`
  - `TEST MACHINE DATASET:(Large) product reviews 50000 sentiment`

- **Small Dataset Machines:**
  - `TEST MACHINE DATASET:(Small) CakeLowButter`
  - `TEST MACHINE DATASET:(Small) Heart Attack Analysis & Prediction`
  - `TEST MACHINE DATASET:(Small) heart_failure_clinical_records`

- **Other Test Dataset Machines:**
  - Various other datasets including census_income, Cleaned_Laptop_data, Communities and Crime Data Set, healthcare-dataset-stroke-data, Hotel Reservations Dataset, House price prediction, Housing value prediction, kickstarter projects, Laptop price datasets, Mobile Price Classification, chocolate bar ratings, sentiment analysis datasets, and more.

## How These Machines Are Used During Training

### 1. Neural Network Configuration Selection

When training a new neural network, the system:

1. **Analyzes the problem characteristics:**
   - Determines the number of inputs and outputs
   - Identifies the appropriate complexity level (Level 1, 2, or 3)

2. **Queries relevant configuration machines:**
   - Searches for machines matching the input/output ranges and level
   - Retrieves experiences and results from similar past configurations

3. **Applies learned knowledge:**
   - Uses successful configurations as starting points
   - Avoids configurations that have historically performed poorly
   - Optimizes hyperparameters based on accumulated experiences

### 2. Feature Engineering Transformer Selection

During feature engineering:

1. **Determines data types:**
   - Identifies whether features are Float, Label, Date, or Time

2. **Selects appropriate budget:**
   - Chooses budget (16, 64, or 256) based on dataset size and computational constraints

3. **Queries FET selector machines:**
   - Retrieves experiences about effective FET combinations for similar data types and budgets
   - Considers which transformer combinations have worked well in the past

4. **Applies optimal transformers:**
   - Uses proven FET combinations to transform features
   - Avoids redundant or ineffective transformer combinations

### 3. Training Optimization

Throughout the training process:

1. **Leverages training cycle results:**
   - Uses `__Results_NNEngine_Training_Cycle__` to optimize individual training cycles
   - Applies lessons learned from previous cycles

2. **References full training results:**
   - Uses `__Results_NNEngine_Training_Full__` to understand complete training patterns
   - Identifies successful training strategies

3. **Finds best configurations:**
   - Uses `__Results_Find_Best_NN_Configuration__` to quickly identify top-performing configurations
   - Reduces search space by focusing on proven architectures

### 4. Continuous Learning

The system continuously updates these machines:

- **Experiences machines** (`__SF_Experiences_*`) store ongoing learning from each training run
- **Results machines** (`__SF_Results_*`) store final outcomes and performance metrics
- As more training runs complete, the machines become more accurate and useful

## Benefits

By using these pre-trained machines:

1. **Faster Training:** The system can start from proven configurations rather than random exploration
2. **Better Performance:** Leverages accumulated knowledge from many previous training runs
3. **Resource Efficiency:** Reduces computational waste by avoiding known poor configurations
4. **Adaptability:** Machines are continuously updated, improving over time
5. **Scalability:** Knowledge from one dataset can inform training on similar datasets

## Note

The list of machines provided here is indicative and may change as the system evolves. New machine types may be added, and existing machines may be updated or refined based on ongoing training experiences.

