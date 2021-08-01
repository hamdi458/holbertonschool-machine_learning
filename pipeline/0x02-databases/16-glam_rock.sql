-- lists all bands with Glam rock as their main style, ranked by their longevity
SELECT band_name, IF(split IS NULL, 2020 - formed, split - formed) AS lifespan
FROM metal_bands WHERE style LIKE '%Glam Rock%'
ORDER BY lifespan DESC;