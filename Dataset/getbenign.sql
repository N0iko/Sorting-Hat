WITH
  contract_deployments AS (
    SELECT
      receipt_contract_address AS contract_address,
      block_number, 
      gas AS deployment_gas,
      nonce AS creator_nonce,
      (LENGTH(input) / 2 - 1) AS bytecode_size,
      block_timestamp AS deploy_time
    FROM
      `bigquery-public-data.crypto_ethereum.transactions`
    WHERE
      block_timestamp >= TIMESTAMP("2022-07-01")
      AND block_timestamp <= TIMESTAMP("2025-12-31")
      AND to_address IS NULL
      AND receipt_status = 1
      AND (LENGTH(input) / 2 - 1) > 500 
  ),
  interaction_metrics AS (
    SELECT
      t.to_address AS contract_address,
      COUNT(DISTINCT t.from_address) AS unique_user_count,
      COUNT(*) AS total_interactions
    FROM
      `bigquery-public-data.crypto_ethereum.transactions` t
    INNER JOIN
      contract_deployments d ON t.to_address = d.contract_address
    WHERE
      t.block_timestamp >= TIMESTAMP("2024-07-01")
      AND t.block_timestamp <= TIMESTAMP("2025-12-31")
      AND t.receipt_status = 1
    GROUP BY t.to_address
    HAVING unique_user_count > 10
  )
SELECT
  m.contract_address,
  d.block_number,
  m.unique_user_count,
  m.total_interactions,
  d.deploy_time,
  d.deployment_gas,
  d.creator_nonce,
  d.bytecode_size
FROM
  interaction_metrics m
JOIN
  contract_deployments d ON m.contract_address = d.contract_address
ORDER BY
  m.unique_user_count DESC
LIMIT 5000;