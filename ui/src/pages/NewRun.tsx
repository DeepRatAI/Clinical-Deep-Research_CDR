import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { createRun } from '../api'
import { ArrowLeft, Send, Loader2, Info } from 'lucide-react'

export default function NewRun() {
  const navigate = useNavigate()
  const [question, setQuestion] = useState('')
  const [maxResults, setMaxResults] = useState(100)
  const [model, setModel] = useState('llama-3.1-8b-instant')
  const [dodLevel, setDodLevel] = useState(1)

  const mutation = useMutation({
    mutationFn: createRun,
    onSuccess: (data) => {
      navigate(`/runs/${data.run_id}`)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate({
      research_question: question,
      max_results: maxResults,
      model,
      output_formats: ['markdown', 'json'],
      dod_level: dodLevel,
    })
  }

  const exampleQuestions = [
    'What is the efficacy of metformin vs sulfonylureas for glycemic control in type 2 diabetes?',
    'Does cognitive behavioral therapy reduce depression symptoms in adolescents compared to waitlist control?',
    'Is low-dose aspirin effective for primary prevention of cardiovascular events in diabetic patients?',
  ]

  return (
    <div className="p-6 max-w-3xl mx-auto">
      {/* Back button */}
      <button
        onClick={() => navigate('/')}
        className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Dashboard
      </button>

      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">New Research Run</h1>
        <p className="text-gray-500 mt-1">
          Enter a clinical research question to start evidence synthesis
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Research question */}
        <div>
          <label
            htmlFor="question"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Research Question
          </label>
          <textarea
            id="question"
            rows={4}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your clinical research question in PICO format..."
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none"
            required
            minLength={10}
            maxLength={2000}
          />
          <p className="mt-1 text-xs text-gray-500">
            Tip: Use PICO format (Population, Intervention, Comparator, Outcome)
          </p>
        </div>

        {/* Example questions */}
        <div className="card">
          <div className="card-header flex items-center gap-2">
            <Info className="w-4 h-4 text-gray-400" />
            <span className="text-sm font-medium text-gray-700">
              Example Questions
            </span>
          </div>
          <div className="card-body space-y-2">
            {exampleQuestions.map((q, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setQuestion(q)}
                className="w-full text-left text-sm text-gray-600 hover:text-primary-600 hover:bg-primary-50 p-2 rounded transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        {/* Advanced options */}
        <details className="card">
          <summary className="card-header cursor-pointer select-none">
            <span className="text-sm font-medium text-gray-700">
              Advanced Options
            </span>
          </summary>
          <div className="card-body space-y-4">
            {/* Max results */}
            <div>
              <label
                htmlFor="maxResults"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Max Results per Source
              </label>
              <input
                type="number"
                id="maxResults"
                value={maxResults}
                onChange={(e) => setMaxResults(Number(e.target.value))}
                min={10}
                max={500}
                className="w-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              />
            </div>

            {/* DoD Level */}
            <div>
              <label
                htmlFor="dodLevel"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                DoD Level
              </label>
              <select
                id="dodLevel"
                value={dodLevel}
                onChange={(e) => setDodLevel(Number(e.target.value))}
                className="w-64 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                <option value={1}>Level 1 - Basic (fast)</option>
                <option value={2}>Level 2 - Full (comprehensive)</option>
                <option value={3}>Level 3 - Research (SOTA)</option>
              </select>
              <p className="mt-1 text-xs text-gray-500">
                Higher levels require more evidence validation and take longer
              </p>
            </div>

            {/* Model */}
            <div>
              <label
                htmlFor="model"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                LLM Model
              </label>
              <select
                id="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-64 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                <optgroup label="Groq (FREE - Recomendado)">
                  <option value="llama-3.1-8b-instant">Llama 3.1 8B Instant (default, fast)</option>
                  <option value="llama-3.3-70b-versatile">Llama 3.3 70B Versatile</option>
                  <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
                </optgroup>
                <optgroup label="HuggingFace (requiere créditos)">
                  <option value="meta-llama/Llama-3.1-70B-Instruct">Llama 3.1 70B</option>
                  <option value="Qwen/Qwen2.5-72B-Instruct">Qwen 2.5 72B</option>
                </optgroup>
              </select>
            </div>
          </div>
        </details>

        {/* Error display */}
        {mutation.error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-700">
              {mutation.error instanceof Error
                ? mutation.error.message
                : 'Failed to create run'}
            </p>
          </div>
        )}

        {/* Submit button */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={mutation.isPending || question.length < 10}
            className="btn btn-primary px-6"
          >
            {mutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Send className="w-4 h-4 mr-2" />
                Start Research Run
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
