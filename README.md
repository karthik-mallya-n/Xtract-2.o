# ML Platform Frontend

A modern, intuitive web application for machine learning workflows built with Next.js and Tailwind CSS.

## Features

### 🏠 Home Page
- Clean, professional landing page
- Clear call-to-action buttons
- Overview of the 4-step ML process

### 📊 Dataset Upload
- Drag and drop file upload interface
- Support for CSV, JSON, and Excel files
- Data type classification (labeled/unlabeled, continuous/categorical)
- Loading states and progress indicators

### 🎯 Model Selection
- AI-powered model recommendations
- Comparison grid of alternative models
- Expected accuracy metrics for each model
- Clear selection interface

### 🚀 Training Status
- Real-time training progress tracking
- Live metrics display (accuracy, precision, recall)
- Training logs sidebar
- Progress bars and completion notifications

### 🔮 Inference
- Dynamic form for model input parameters
- Real-time prediction results
- Confidence scores and probability metrics
- Prediction history tracking
- Model information display

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Font**: Inter (Google Fonts)

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

# Machine Learning Platform

A comprehensive full-stack machine learning platform built with Next.js frontend and Flask backend. This platform allows users to upload datasets, get AI-powered model recommendations, train models, and make predictions.

## 🚀 Project Structure

```
📁 Root Directory/
├── 📁 src/                          # Next.js Frontend
│   ├── 📁 app/                      # App Router pages
│   │   ├── page.tsx                 # Home page
│   │   ├── upload/page.tsx          # Dataset upload
│   │   ├── select-model/page.tsx    # Model selection
│   │   ├── training-status/page.tsx # Training progress
│   │   └── inference/page.tsx       # Model predictions
│   └── 📁 components/               # Reusable UI components
│       ├── navigation.tsx           # Main navigation
│       ├── button.tsx               # Custom button
│       ├── card.tsx                 # Card component
│       ├── badge.tsx                # Status badges
│       ├── loading-spinner.tsx      # Loading indicator
│       ├── progress-bar.tsx         # Progress visualization
│       └── alert.tsx                # Alert messages
├── 📁 my_flask_app/                 # Flask Backend
│   ├── app.py                       # Main Flask application
│   ├── core_ml.py                   # ML core functionality
│   ├── tasks.py                     # Celery background tasks
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Environment variables
│   ├── start_backend.bat           # Windows startup script
│   ├── start_celery.bat            # Celery worker script
│   └── test_api.py                 # API testing script
├── package.json                     # Node.js dependencies
├── tailwind.config.ts              # Tailwind CSS config
├── tsconfig.json                   # TypeScript config
└── README.md                       # This file
```

## 🎯 Features

### Frontend (Next.js + TypeScript + Tailwind CSS)
- **Home Page**: Landing page with platform overview
- **Dataset Upload**: Drag-and-drop file upload with data classification
- **Model Selection**: AI-powered model recommendations
- **Training Status**: Real-time training progress tracking
- **Inference**: Interactive prediction interface
- **Responsive Design**: Mobile-friendly UI
- **Type Safety**: Full TypeScript implementation

### Backend (Flask + Python + ML Libraries)
- **File Upload API**: Secure dataset upload and validation
- **Dataset Analysis**: Automatic data profiling and insights
- **LLM Integration**: OpenRouter API for model recommendations
- **Model Training**: Scikit-learn integration with multiple algorithms
- **Background Tasks**: Celery for async model training
- **Prediction API**: RESTful endpoints for model inference
- **Model Storage**: Persistent model storage with joblib

## 🛠 Technologies Used

### Frontend
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icons

### Backend
- **Flask** - Python web framework
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning library
- **Celery** - Distributed task queue
- **Redis** - Message broker for Celery
- **OpenRouter** - LLM API integration
- **Joblib** - Model persistence

## 📦 Installation & Setup

### Prerequisites
- **Node.js** (v18 or higher)
- **Python** (v3.8 or higher)
- **Redis Server** (for background tasks)

### Frontend Setup

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```

3. **Access the frontend:**
   ```
   http://localhost:3000
   ```

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd my_flask_app
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Update `OPENROUTER_API_KEY` with your API key

4. **Start Redis server:**
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis:latest
   
   # Or install Redis locally
   # Windows: Download from https://redis.io/download
   # macOS: brew install redis
   # Ubuntu: sudo apt-get install redis-server
   ```

5. **Start the Flask server:**
   ```bash
   # Windows
   .\start_backend.bat
   
   # Or manually
   python app.py
   ```

6. **Start Celery worker (in another terminal):**
   ```bash
   # Windows
   .\start_celery.bat
   
   # Or manually
   celery -A tasks worker --loglevel=info
   ```

## 🔌 API Endpoints

### Core Endpoints
- `GET /api/health` - Health check
- `POST /api/upload` - Upload dataset file
- `GET /api/recommend-model` - Get model recommendations
- `POST /api/train` - Start model training
- `GET /api/training-status/<task_id>` - Check training progress
- `POST /api/predict` - Make predictions
- `GET /api/models` - List trained models

### Request/Response Examples

#### Upload Dataset
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@dataset.csv" \
  -F "is_labeled=labeled" \
  -F "data_type=categorical"
```

#### Get Model Recommendations
```bash
curl "http://localhost:5000/api/recommend-model?file_id=<file_id>"
```

#### Start Training
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"file_id": "<file_id>", "model_name": "RandomForest"}'
```

#### Make Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"file_id": "<file_id>", "input_data": {"feature1": "value1"}}'
```

## 🧪 Testing

### Test Backend API
```bash
cd my_flask_app
python test_api.py
```

### Test Frontend
```bash
npm test
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `my_flask_app` directory:

```env
# Flask Configuration
FLASK_DEBUG=True
FLASK_PORT=5000
SECRET_KEY=your-secret-key-here
MAX_CONTENT_LENGTH=16777216

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# File Storage
UPLOAD_FOLDER=uploads
MODEL_STORAGE_PATH=models

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## 🤖 Machine Learning Features

### Supported Algorithms
- **Random Forest** - Ensemble learning
- **Support Vector Machine** - Classification/Regression
- **Gradient Boosting** - Boosting ensemble
- **Logistic Regression** - Linear classification
- **Decision Tree** - Tree-based learning

### Data Processing
- **Automatic type detection** - Numeric vs categorical
- **Missing value handling** - Intelligent imputation
- **Feature scaling** - StandardScaler normalization
- **Label encoding** - Categorical variable encoding
- **Train/test splitting** - Automated data splitting

### Model Evaluation
- **Accuracy** - Overall prediction accuracy
- **Precision** - Positive prediction accuracy
- **Recall** - True positive detection rate
- **F1-Score** - Harmonic mean of precision/recall
- **Confusion Matrix** - Detailed classification results

## 🎨 UI Components

### Custom Components
- **Navigation** - Responsive navigation bar
- **Button** - Styled button with variants
- **Card** - Flexible content containers
- **Badge** - Status indicators
- **Progress Bar** - Training progress visualization
- **Loading Spinner** - Loading states
- **Alert** - Success/error messages

### Design System
- **Color Palette** - Consistent brand colors
- **Typography** - Tailwind typography scale
- **Spacing** - Consistent padding/margins
- **Responsive Design** - Mobile-first approach

## 🚦 Workflow

1. **Upload Dataset** - Users upload CSV/Excel files
2. **Data Analysis** - Backend analyzes dataset structure
3. **Model Recommendation** - LLM suggests optimal models
4. **Model Training** - Background training with progress tracking
5. **Model Evaluation** - Performance metrics calculation
6. **Prediction** - Interactive prediction interface
7. **Model Management** - Save and reuse trained models

## 🔒 Security Features

- **File validation** - Secure file type checking
- **Input sanitization** - Secure filename handling
- **CORS protection** - Cross-origin request security
- **Environment variables** - Secure configuration management
- **Error handling** - Graceful error responses

## 📈 Performance

- **Async processing** - Non-blocking model training
- **Caching** - Redis-based result caching
- **Lazy loading** - Optimized component loading
- **Code splitting** - Next.js automatic optimization
- **Compression** - Gzip response compression

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Flask server won't start:**
- Check if port 5000 is available
- Verify Python dependencies are installed
- Check `.env` file configuration

**Celery worker not connecting:**
- Ensure Redis server is running
- Check Redis connection settings
- Verify Celery is installed

**Frontend build errors:**
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall
- Check Node.js version compatibility

**API connection issues:**
- Verify CORS configuration
- Check API endpoint URLs
- Ensure backend is running on correct port

### Performance Optimization

**Large dataset handling:**
- Increase `MAX_CONTENT_LENGTH` in .env
- Consider chunked file upload
- Use data sampling for large files

**Training speed:**
- Reduce dataset size for testing
- Use faster algorithms (e.g., Logistic Regression)
- Implement model parameter optimization

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

---

**Built with ❤️ using modern web technologies**

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
