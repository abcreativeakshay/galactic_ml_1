/*
  # Fix Database Security and Performance Issues

  ## Changes Made

  ### 1. Add Missing Foreign Key Indexes
  - Add index on `generated_notebooks.session_id` for FK lookup performance
  - Add index on `generation_sessions.dataset_id` for FK lookup performance

  ### 2. Remove Unused Indexes
  Dropping 8 unused indexes to improve write performance and reduce storage:
  - idx_ml_models_experiment_id
  - idx_ml_metrics_model_id
  - idx_hyperparameter_tuning_runs_model_id
  - idx_ml_experiments_created_at
  - idx_datasets_user_id
  - idx_sessions_user_id
  - idx_sessions_status
  - idx_logs_session_id

  ### 3. Fix Multiple Permissive Policies
  Remove overly permissive public policies:
  - Drop "Allow public" policies on `generated_notebooks` (read, insert, update)
  - Drop "Allow public" policies on `generation_logs` (read, insert)
  - Keep only the secure user-based policies that check authentication

  ### 4. Fix Function Security
  - Set immutable search_path on `update_updated_at_column` function

  ## Security Impact
  - Removes overly permissive public access policies
  - Ensures all data access is properly authenticated
  - Prevents SQL injection via search_path manipulation
*/

-- =====================================================
-- 1. ADD MISSING FOREIGN KEY INDEXES
-- =====================================================

-- Add index for generated_notebooks.session_id FK
CREATE INDEX IF NOT EXISTS idx_generated_notebooks_session_id 
ON public.generated_notebooks(session_id);

-- Add index for generation_sessions.dataset_id FK
CREATE INDEX IF NOT EXISTS idx_generation_sessions_dataset_id 
ON public.generation_sessions(dataset_id);

-- =====================================================
-- 2. DROP UNUSED INDEXES
-- =====================================================

DROP INDEX IF EXISTS public.idx_ml_models_experiment_id;
DROP INDEX IF EXISTS public.idx_ml_metrics_model_id;
DROP INDEX IF EXISTS public.idx_hyperparameter_tuning_runs_model_id;
DROP INDEX IF EXISTS public.idx_ml_experiments_created_at;
DROP INDEX IF EXISTS public.idx_datasets_user_id;
DROP INDEX IF EXISTS public.idx_sessions_user_id;
DROP INDEX IF EXISTS public.idx_sessions_status;
DROP INDEX IF EXISTS public.idx_logs_session_id;

-- =====================================================
-- 3. FIX MULTIPLE PERMISSIVE POLICIES
-- =====================================================

-- Drop overly permissive public policies on generated_notebooks
DROP POLICY IF EXISTS "Allow public read access to notebooks" ON public.generated_notebooks;
DROP POLICY IF EXISTS "Allow public insert to notebooks" ON public.generated_notebooks;
DROP POLICY IF EXISTS "Allow public update to notebooks" ON public.generated_notebooks;

-- Drop overly permissive public policies on generation_logs
DROP POLICY IF EXISTS "Allow public read access to logs" ON public.generation_logs;
DROP POLICY IF EXISTS "Allow public insert to logs" ON public.generation_logs;

-- The secure user-based policies already exist and will remain:
-- - "Users can access their generated notebooks" (SELECT, INSERT, UPDATE)
-- - "Users can view their generation logs" (SELECT, INSERT)

-- =====================================================
-- 4. FIX FUNCTION SECURITY (IMMUTABLE SEARCH_PATH)
-- =====================================================

-- Recreate the update_updated_at_column function with secure search_path
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION public.update_updated_at_column() TO authenticated;
GRANT EXECUTE ON FUNCTION public.update_updated_at_column() TO service_role;
