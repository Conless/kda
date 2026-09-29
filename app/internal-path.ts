const repositoryName = process.env.GITHUB_REPOSITORY?.split('/')[1];
const basePath = process.env.GITHUB_PAGES === 'true' && repositoryName
  ? `/${repositoryName}`
  : '';

export function internalPath(path: string) {
  return `${basePath}${path}`;
}
