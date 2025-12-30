/*
  # Notebook Generation System Schema

  1. New Tables
    - `generated_notebooks`
      - `id` (uuid, primary key)
      - `user_id` (text) - For future auth integration
      - `dataset_name` (text)
      - `dataset_info` (jsonb) - Stores dataset metadata
      - `problem_type` (text) - classification/regression/clustering
      - `target_column` (text)
      - `selected_models` (text[])
      - `notebook_content` (text) - Generated notebook JSON
      - `theory_content` (jsonb) - AI-generated theory sections
      - `status` (text) - pending/generating/completed/failed
      - `created_at` (timestamptz)
      - `updated_at` (timestamptz)
    
    - `generation_logs`
      - `id` (uuid, primary key)
      - `notebook_id` (uuid, foreign key)
      - `step` (text)
      - `status` (text)
      - `message` (text)
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on both tables
    - Public access for demo (can be restricted later with auth)
*/

CREATE TABLE IF NOT EXISTS generated_notebooks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text DEFAULT 'anonymous',
  dataset_name text NOT NULL,
  dataset_info jsonb DEFAULT '{}'::jsonb,
  problem_type text,
  target_column text,
  selected_models text[],
  notebook_content text,
  theory_content jsonb DEFAULT '{}'::jsonb,
  status text DEFAULT 'pending',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS generation_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  notebook_id uuid REFERENCES generated_notebooks(id) ON DELETE CASCADE,
  step text NOT NULL,
  status text NOT NULL,
  message text,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE generated_notebooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE generation_logs ENABLE ROW LEVEL SECURITY;

-- Allow public read/write for demo purposes
CREATE POLICY "Allow public read access to notebooks"
  ON generated_notebooks FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public insert to notebooks"
  ON generated_notebooks FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Allow public update to notebooks"
  ON generated_notebooks FOR UPDATE
  TO public
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Allow public read access to logs"
  ON generation_logs FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public insert to logs"
  ON generation_logs FOR INSERT
  TO public
  WITH CHECK (true);
