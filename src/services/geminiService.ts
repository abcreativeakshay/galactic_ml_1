const GEMINI_API_KEY = 'AIzaSyAcK2RJcvfDsEm0na4G65O2ucP4Ial2c-0';
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent';

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
  }>;
}

export async function generateTheoryContent(modelName: string, context: string): Promise<string> {
  const prompt = `Generate a comprehensive theory section for the ${modelName} machine learning model. Include:

1. **Mathematical Foundation**: Core equations and principles
2. **Algorithm Explanation**: How the model works step-by-step
3. **Key Parameters**: Important hyperparameters and their effects
4. **Use Cases**: When to use this model vs alternatives
5. **Advantages**: Strengths of this approach
6. **Limitations**: Weaknesses and considerations
7. **Best Practices**: Tips for optimal performance

Context: ${context}

Format the response in clear markdown with tables where appropriate. Be concise but comprehensive.`;

  try {
    const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: prompt
          }]
        }]
      })
    });

    if (!response.ok) {
      throw new Error(`Gemini API error: ${response.statusText}`);
    }

    const data: GeminiResponse = await response.json();
    return data.candidates[0]?.content?.parts[0]?.text || 'Theory generation failed';
  } catch (error) {
    console.error('Error generating theory:', error);
    return `# ${modelName} Theory\n\nError generating theory content. Using default template.`;
  }
}

export async function generateCheatSheet(category: string): Promise<string> {
  const prompt = `Create a comprehensive cheat sheet table for ${category} in machine learning.

Include:
- Technique names
- When to use
- Key parameters
- Code snippets (Python)
- Common pitfalls

Format as a markdown table that is easy to read and reference.`;

  try {
    const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: prompt
          }]
        }]
      })
    });

    if (!response.ok) {
      throw new Error(`Gemini API error: ${response.statusText}`);
    }

    const data: GeminiResponse = await response.json();
    return data.candidates[0]?.content?.parts[0]?.text || 'Cheat sheet generation failed';
  } catch (error) {
    console.error('Error generating cheat sheet:', error);
    return `Error generating ${category} cheat sheet.`;
  }
}

export async function analyzeDataset(datasetSample: any[]): Promise<{
  problemType: string;
  suggestedTarget: string[];
  recommendedModels: string[];
}> {
  const prompt = `Analyze this dataset sample and provide:
1. Problem type (classification/regression/clustering)
2. Suggested target columns (list potential target variables)
3. Recommended ML models (5-7 models that would work well)

Dataset sample (first 5 rows):
${JSON.stringify(datasetSample, null, 2)}

Respond in JSON format:
{
  "problemType": "classification|regression|clustering",
  "suggestedTarget": ["column1", "column2"],
  "recommendedModels": ["model1", "model2"]
}`;

  try {
    const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: prompt
          }]
        }]
      })
    });

    if (!response.ok) {
      throw new Error(`Gemini API error: ${response.statusText}`);
    }

    const data: GeminiResponse = await response.json();
    const text = data.candidates[0]?.content?.parts[0]?.text || '{}';

    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }

    return {
      problemType: 'classification',
      suggestedTarget: [],
      recommendedModels: ['Random Forest', 'XGBoost', 'Logistic Regression']
    };
  } catch (error) {
    console.error('Error analyzing dataset:', error);
    return {
      problemType: 'classification',
      suggestedTarget: [],
      recommendedModels: ['Random Forest', 'XGBoost', 'Logistic Regression']
    };
  }
}
